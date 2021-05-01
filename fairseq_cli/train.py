#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import random
import sys

import numpy as np
import torch
from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
import signal
import socket

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")


# define the handler function
# note that this is not executed here, but rather
# when the associated signal is sent
def sig_handler(signum, frame):
    print("caught signal", signum)
    print(socket.gethostname(), "USR1 signal caught.")
    # do other stuff to cleanup here
    print('requeuing job ' + os.environ['SLURM_JOB_ID'])
    os.system('scontrol requeue ' + os.environ['SLURM_JOB_ID'])
    sys.exit(-1)


def term_handler(signum, frame):
    print("bypassing sigterm", flush=True)


def main(args):
    utils.import_user_module(args)

    assert (
        args.max_tokens is not None or args.max_sentences is not None
    ), "Must specify batch size either with --max-tokens or --max-sentences"

    metrics.reset()

    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    logger.info(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    logger.info(model)
    logger.info("task: {} ({})".format(args.task, task.__class__.__name__))
    logger.info("model: {} ({})".format(args.arch, model.__class__.__name__))
    logger.info(
        "criterion: {} ({})".format(args.criterion, criterion.__class__.__name__)
    )
    logger.info(
        "num. model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )

    # (optionally) Configure quantization
    if args.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=args.quantization_config_path,
            max_epoch=args.max_epoch,
            max_update=args.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if args.model_parallel_size == 1:
        trainer = Trainer(args, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(args, task, model, criterion)

    logger.info(
        "training on {} devices (GPUs/TPUs)".format(args.distributed_world_size)
    )
    logger.info(
        "max tokens per GPU = {} and max sentences per GPU = {}".format(
            args.max_tokens, args.max_sentences
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        args,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()

    while lr > args.min_lr and epoch_itr.next_epoch_idx <= max_epoch:
        # train for one epoch
        valid_losses, should_stop = train(args, trainer, task, epoch_itr)

        if hasattr(args, 'compute_train_dynamics') and args.compute_train_dynamics:
            if hasattr(args, 'analyze') and args.analyze:
                analysis_on_the_fly_train_dynamics(args, trainer, task, epoch_itr)
            elif args.competent_cl:
                # warmup = ERM
                competent_cl_on_the_fly_train_dynamics(args, trainer, task, epoch_itr)
            elif hasattr(args, 'burnout_epochs') and args.burnout_epochs > 0:
                # constantly update train dynamics
                # warmup = ERM; burnout = data selection
                on_the_fly_train_dynamics(args, trainer, task, epoch_itr)
            else:
                # use stale train dynamics from the warmup epochs
                # warmup = ERM, data selection
                fix_on_the_fly_train_dynamics(args, trainer, task, epoch_itr)

        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))


def should_stop_early(args, valid_loss):
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= args.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    args.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > args.curriculum),
    )
    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    if getattr(args, "tpu", False):
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),
        default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
    )

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = args.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % args.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            args, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def validate_and_save(args, trainer, task, epoch_itr, valid_subsets, end_of_epoch):
    num_updates = trainer.get_num_updates()
    max_update = args.max_update or math.inf
    do_save = (
        (end_of_epoch and epoch_itr.epoch % args.save_interval == 0)
        or num_updates >= max_update
        or (
            args.save_interval_updates > 0
            and num_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates >= args.validate_after_updates
        )
    )
    do_validate = (
        (not end_of_epoch and do_save)  # validate during mid-epoch saves
        or (end_of_epoch and epoch_itr.epoch % args.validate_interval == 0)
        or num_updates >= max_update
        or (
            args.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % args.validate_interval_updates == 0
        )
    ) and not args.disable_validation

    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

    # Stopping conditions
    should_stop = (
        should_stop_early(args, valid_losses[0])
        or num_updates >= max_update
        or (
            args.stop_time_hours > 0
            and trainer.cumulative_training_time() / (60 * 60) > args.stop_time_hours
        )
    )

    # Save checkpoint
    if do_save or should_stop:
        logger.info("begin save checkpoint")
        checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

    return valid_losses, should_stop


def get_training_stats(stats):
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    valid_losses = []
    for subset in subsets:
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(shuffle=False)
        if getattr(args, "tpu", False):
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                args.tensorboard_logdir if distributed_utils.is_master(args) else None
            ),
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
        if "fg_gloss0" in stats:
            criterion = trainer.get_criterion()
            ngroups = criterion.n_groups
            baselines = torch.zeros(ngroups, device='cuda')
            for ii in range(ngroups):
                key = "fg_gloss{}".format(ii)
                baselines[ii] = stats[key]
                stats.pop(key, None)
            if hasattr(criterion, 'set_valid_baselines'):
                criterion.set_valid_baselines(baselines)

        progress.print(stats, tag=subset, step=trainer.get_num_updates())
        valid_losses.append(stats[args.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(args, trainer, stats):
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best, stats[args.best_checkpoint_metric]
        )
    return stats


def competent_cl_on_the_fly_train_dynamics(args, trainer, task, epoch_itr):
    if hasattr(args, 'warmup_epochs') and epoch_itr.epoch < args.warmup_epochs:
        return

    def competence_func(t, T):
        return min(1, math.sqrt(t * (1 - 0.3**2) / T + 0.3**2))

    ratio = competence_func(trainer.get_num_updates(), 200000)
    if ratio >= 1:
        return

    cur_subset = 'concat_train'
    data_size = len(trainer.task.datasets[cur_subset])
    train_hardness = torch.zeros(data_size, device='cuda')
    sanity_ids = torch.zeros(data_size, device='cuda')

    itr = task.get_batch_iterator(
            dataset=trainer.task.dataset(cur_subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                trainer.task.max_positions(),
                trainer.model.max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=trainer.data_parallel_world_size,
            shard_id=trainer.data_parallel_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)

    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=10000,
        epoch=epoch_itr.epoch,
        prefix=f"summarize on train subset",
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),
        default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
    )
    # create a new root metrics aggregator so validation metrics
    # don't pollute other aggregators (e.g., train meters)
    with metrics.aggregate(new_root=True) as agg:
        for i, sample in enumerate(progress):
            log_output, sample_ids, b_train_hardness, is_dummy = trainer.compute_train_dynamics_step(sample)
            if not is_dummy:
                sanity_ids[sample_ids] = 1
                train_hardness[sample_ids] = b_train_hardness

            if i % 500 == 0:
                # log stats
                stats = agg.get_smoothed_values()
                progress.print(stats, tag=cur_subset, step=i)

    torch.distributed.all_reduce(sanity_ids)
    diff = (sanity_ids - torch.ones(data_size, device='cuda')).sum().item()
    assert diff == 0, "diff={}".format(diff)
    torch.distributed.all_reduce(train_hardness)

    def _write_to_file(nparray):
        fout = os.path.join(args.save_dir, "hardness_last.npy")
        if distributed_utils.is_master(args):
            if os.path.exists(fout):
                os.remove(fout)
            np.save(fout, nparray)

    train_hardness = train_hardness.cpu().numpy()
    _write_to_file(train_hardness)
    logger.info("saved files to the disk for epoch = {}".format(epoch_itr.epoch))
    logger.info("epoch = {}, CL ratio = {}".format(epoch_itr.epoch, ratio))
    with open(os.path.join(args.save_dir, "cl_ratio"), "w") as fout:
        fout.write("{}".format(ratio))
    trainer.task.datasets['train'].set_data_properties(train_hardness, None, ratio)


def on_the_fly_train_dynamics(args, trainer, task, epoch_itr):
    cur_subset = 'concat_train'
    data_size = len(trainer.task.datasets[cur_subset])
    # train_avg_probs = torch.zeros(data_size, device='cuda')
    train_med_probs = torch.zeros(data_size, device='cuda')
    # train_avg_ent = torch.zeros(data_size, device='cuda')
    sanity_ids = torch.zeros(data_size, device='cuda')

    itr = task.get_batch_iterator(
            dataset=trainer.task.dataset(cur_subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                trainer.task.max_positions(),
                trainer.model.max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=trainer.data_parallel_world_size,
            shard_id=trainer.data_parallel_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)

    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=10000,
        epoch=epoch_itr.epoch,
        prefix=f"summarize on train subset",
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),
        default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
    )
    # create a new root metrics aggregator so validation metrics
    # don't pollute other aggregators (e.g., train meters)
    with metrics.aggregate(new_root=True) as agg:
        for i, sample in enumerate(progress):
            log_output, sample_ids, median_p, is_dummy = trainer.compute_train_dynamics_step(sample)
            if not is_dummy:
                sanity_ids[sample_ids] = 1
                # train_avg_probs[sample_ids] = average_p
                train_med_probs[sample_ids] = median_p
                # train_avg_ent[sample_ids] = avg_ent

            if i % 500 == 0:
                # log stats
                stats = agg.get_smoothed_values()
                progress.print(stats, tag=cur_subset, step=i)

    torch.distributed.all_reduce(sanity_ids)
    diff = (sanity_ids - torch.ones(data_size, device='cuda')).sum().item()
    assert diff == 0, "diff={}".format(diff)
    # torch.distributed.all_reduce(train_avg_probs)
    torch.distributed.all_reduce(train_med_probs)
    # torch.distributed.all_reduce(train_avg_ent)

    def _write_to_file(tensor, fname):
        tensor = tensor.cpu().numpy()
        fout = os.path.join(args.save_dir, "{}_{}.npy".format(fname, epoch_itr.epoch))
        np.save(fout, tensor)

    def _get_confidence_and_variability(epochs_of_vecs):
        mat = np.vstack(epochs_of_vecs)
        mu = np.mean(mat, axis=0)
        var = np.std(mat, axis=0)
        return mu, var
    # _write_to_file(train_avg_probs, "avg_probs")
    if distributed_utils.is_master(args):
        _write_to_file(train_med_probs, "med_probs")
    # _write_to_file(train_avg_ent, "avg_ent")
    logger.info("saved files to the disk for epoch = {}".format(epoch_itr.epoch))

    if epoch_itr.epoch >= args.burnout_epochs:
        med_probs = []
        for eid in range(2, epoch_itr.epoch+1):
            path = os.path.join(args.save_dir, "med_probs_{}.npy".format(eid))
            if not os.path.exists(path):
                continue
            med_probs.append(np.load(path))
        mu, var = _get_confidence_and_variability(med_probs)
        trainer.task.datasets['train'].set_data_properties(mu, var)


def fix_on_the_fly_train_dynamics(args, trainer, task, epoch_itr):
    if hasattr(args, 'warmup_epochs') and epoch_itr.epoch > args.warmup_epochs:
        return

    cur_subset = 'concat_train'
    data_size = len(trainer.task.datasets[cur_subset])
    # train_avg_probs = torch.zeros(data_size, device='cuda')
    train_med_probs = torch.zeros(data_size, device='cuda')
    # train_avg_ent = torch.zeros(data_size, device='cuda')
    sanity_ids = torch.zeros(data_size, device='cuda')

    itr = task.get_batch_iterator(
            dataset=trainer.task.dataset(cur_subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                trainer.task.max_positions(),
                trainer.model.max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=trainer.data_parallel_world_size,
            shard_id=trainer.data_parallel_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)

    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=10000,
        epoch=epoch_itr.epoch,
        prefix=f"summarize on train subset",
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),
        default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
    )
    # create a new root metrics aggregator so validation metrics
    # don't pollute other aggregators (e.g., train meters)
    with metrics.aggregate(new_root=True) as agg:
        for i, sample in enumerate(progress):
            log_output, sample_ids, median_p, is_dummy = trainer.compute_train_dynamics_step(sample)
            if not is_dummy:
                sanity_ids[sample_ids] = 1
                # train_avg_probs[sample_ids] = average_p
                train_med_probs[sample_ids] = median_p
                # train_avg_ent[sample_ids] = avg_ent

            if i % 500 == 0:
                # log stats
                stats = agg.get_smoothed_values()
                progress.print(stats, tag=cur_subset, step=i)

    torch.distributed.all_reduce(sanity_ids)
    diff = (sanity_ids - torch.ones(data_size, device='cuda')).sum().item()
    assert diff == 0, "diff={}".format(diff)
    # torch.distributed.all_reduce(train_avg_probs)
    torch.distributed.all_reduce(train_med_probs)
    # torch.distributed.all_reduce(train_avg_ent)

    def _write_to_file(tensor, fname):
        tensor = tensor.cpu().numpy()
        fout = os.path.join(args.save_dir, "{}_{}.npy".format(fname, epoch_itr.epoch))
        np.save(fout, tensor)

    def _get_confidence_and_variability(epochs_of_vecs):
        mat = np.vstack(epochs_of_vecs)
        mu = np.mean(mat, axis=0)
        var = np.std(mat, axis=0)
        return mu, var
    # _write_to_file(train_avg_probs, "avg_probs")
    if distributed_utils.is_master(args):
        _write_to_file(train_med_probs, "med_probs")
    # _write_to_file(train_avg_ent, "avg_ent")
    logger.info("saved files to the disk for epoch = {}".format(epoch_itr.epoch))

    if epoch_itr.epoch == args.warmup_epochs:
        med_probs = []
        for eid in range(2, epoch_itr.epoch+1):
            path = os.path.join(args.save_dir, "med_probs_{}.npy".format(eid))
            if not os.path.exists(path):
                continue
            med_probs.append(np.load(path))
        mu, var = _get_confidence_and_variability(med_probs)
        trainer.task.datasets['train'].set_data_properties(mu, var)


def analysis_on_the_fly_train_dynamics(args, trainer, task, epoch_itr):
    cur_subset = 'concat_train'
    data_size = len(trainer.task.datasets[cur_subset])

    results = None
    sanity_ids = torch.zeros(data_size, device='cuda')

    itr = task.get_batch_iterator(
            dataset=trainer.task.dataset(cur_subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                trainer.task.max_positions(),
                trainer.model.max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=trainer.data_parallel_world_size,
            shard_id=trainer.data_parallel_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)

    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=10000,
        epoch=epoch_itr.epoch,
        prefix=f"summarize on train subset",
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),
        default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
    )
    # create a new root metrics aggregator so validation metrics
    # don't pollute other aggregators (e.g., train meters)
    with metrics.aggregate(new_root=True) as agg:
        for i, sample in enumerate(progress):
            log_output, sample_ids, return_results, is_dummy = trainer.compute_train_dynamics_step(sample)
            if results is None:
                results = {key: torch.zeros(data_size, device='cuda') for key in return_results.keys()}

            if not is_dummy:
                sanity_ids[sample_ids] = 1
                for key, vec in return_results.items():
                    results[key][sample_ids] = vec

            if i % 500 == 0:
                # log stats
                stats = agg.get_smoothed_values()
                progress.print(stats, tag=cur_subset, step=i)

    torch.distributed.all_reduce(sanity_ids)
    diff = (sanity_ids - torch.ones(data_size, device='cuda')).sum().item()
    assert diff == 0, "diff={}".format(diff)
    for key, value in results.items():
        torch.distributed.all_reduce(results[key])

    def _write_to_file(tensor, fname):
        tensor = tensor.cpu().numpy()
        fout = os.path.join(args.save_dir, "{}_{}.npy".format(fname, epoch_itr.epoch))
        np.save(fout, tensor)

    if distributed_utils.is_master(args):
        for key, value in results.items():
            _write_to_file(value, key)
    logger.info("saved files to the disk for epoch = {}".format(epoch_itr.epoch))


def cli_main(modify_parser=None):
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(args, main)
    else:
        distributed_utils.call_main(args, main)


if __name__ == "__main__":
    signal.signal(signal.SIGUSR1, sig_handler)
    signal.signal(signal.SIGTERM, term_handler)
    print('signal installed', flush=True)

    cli_main()
