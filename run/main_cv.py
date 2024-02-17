import collections
import logging
import multiprocessing
import os
import signal
import sys
import wandb

from src.utils.utils import (
    MyLightningCLI, 
    TrainerWandb, 
    ModelCheckpointNoSave, 
    TempSetContextManager
)


TIMEOUT_S = 1200

# Set up logging
logging.basicConfig(
    level=logging.ERROR,
    format = '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s'
)


Worker = collections.namedtuple("Worker", ("queue", "process"))
WorkerInitData = collections.namedtuple(
    "WorkerInitData", ("fold_index", "sweep_id", "sweep_run_name", "config")
)
WorkerDoneData = collections.namedtuple("WorkerDoneData", ("scores", "fold_index"))


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def train(sweep_q, worker_q):
    reset_wandb_env()
    worker_data = worker_q.get()
    run_name = "{}-{}".format(worker_data.sweep_run_name, worker_data.fold_index)
    config = worker_data.config
    run = wandb.init(
        group=worker_data.sweep_id,
        job_type=worker_data.sweep_run_name,
        name=run_name,
        config=config,
    )

    args = sys.argv[1:] + ['--data.init_args.fold_index', f'{worker_data.fold_index}']
    scores = dict()
    try:
        with TempSetContextManager(sys, 'argv', sys.argv[:1]):
            cli = MyLightningCLI(
                trainer_class=TrainerWandb, 
                save_config_kwargs={
                    'config_filename': 'config_pl.yaml',
                    'overwrite': True,
                },
                args=args,
                run=True,
            )

        # Best scores
        for cb in cli.trainer.checkpoint_callbacks:
            scores[cb.monitor] = cb.best_model_score.item()

        # Cross scores
        for cb_best in cli.trainer.checkpoint_callbacks:
            if not isinstance(cb_best, ModelCheckpointNoSave):
                continue
            best_epoch: int = cb_best.best_epoch
            for cb in cli.trainer.checkpoint_callbacks:
                if cb is cb_best:
                    continue
                if not isinstance(cb_best, ModelCheckpointNoSave):
                    continue

                metric_value = cb.ith_epoch_score(best_epoch)

                if metric_value is not None:
                    scores[f'{cb_best.monitor}_cross_{cb.monitor}'] = \
                        metric_value.item()
                else:
                    scores[f'{cb_best.monitor}_cross_{cb.monitor}'] = None
    except Exception as e:
        print(e)

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(60)  # 60 seconds to finish up

    try:
        run.log(scores)
        wandb.join()
        sweep_q.put(WorkerDoneData(scores=scores, fold_index=worker_data.fold_index))
    except Exception as e:
        print(e)
    signal.alarm(0)
    exit()


def main():
    # Parse args
    args = sys.argv[1:]
    with TempSetContextManager(sys, 'argv', sys.argv[:1]):
        cli = MyLightningCLI(
            trainer_class=TrainerWandb, 
            save_config_kwargs={
                'config_filename': 'config_pl.yaml',
                'overwrite': True,
            },
            args=[arg for arg in args if arg != 'fit'],
            run=False,
        )

    # Spin up workers before calling wandb.init()
    # Workers will be blocked on a queue waiting to start
    sweep_q = multiprocessing.Queue()
    workers = []
    for fold_index in range(cli.config.data.init_args.k):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=train, kwargs=dict(sweep_q=sweep_q, worker_q=q)
        )
        p.start()
        workers.append(Worker(queue=q, process=p))

    sweep_run = wandb.init()
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = "{}/groups/{}".format(project_url, sweep_id)
    sweep_run.notes = sweep_group_url
    sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown"

    # Start CV
    for fold_index in range(cli.config.data.init_args.k):
        worker = workers[fold_index]
        # start worker
        worker.queue.put(
            WorkerInitData(
                sweep_id=sweep_id,
                fold_index=fold_index,
                sweep_run_name=sweep_run_name,
                config=dict(sweep_run.config),
            )
        )

    # Collect results
    scores = collections.defaultdict(dict)
    for _ in range(cli.config.data.init_args.k):
        # get metric from worker
        result = sweep_q.get(timeout=TIMEOUT_S)
        # wait for worker to finish
        worker.process.join(TIMEOUT_S)
        if worker.process.is_alive():
            print("Worker timed out")
            worker.process.terminate()
            worker.process.join()
        try:
            # collect metric to dict & log metric to sweep_run
            for name, value in result.scores.items():
                scores[name][result.fold_index] = value
                sweep_run.log(
                    {
                        f'best_{name}': value,
                        "fold_index": result.fold_index,
                    }
                )
        except Exception as e:
            print(e)

    # Log mean of metrics
    scores_mean = {
        f'mean_best_{name}': 
            sum(value for value in fold_index_to_score.values() if value is not None) / \
            sum(1 for value in fold_index_to_score.values() if value is not None) 
        for name, fold_index_to_score in scores.items()
        if sum(1 for value in fold_index_to_score.values() if value is not None) > 0
    }


    signal.signal(signal.SIGALRM, handler)
    signal.alarm(60)  # 60 seconds to finish up

    try:
        sweep_run.log(scores_mean)
        wandb.join()
    except Exception as e:
        print(e)

    signal.alarm(0)

    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)


# https://stackoverflow.com/questions/492519/timeout-on-a-function-call
# Register an handler for the timeout
def handler(signum, frame):
    print("Forever is over!")
    raise Exception("end of time")


if __name__ == "__main__":
    try:
        main()
    except Exception as e: 
        print(e)
