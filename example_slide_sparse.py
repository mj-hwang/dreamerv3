def main():

  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['medium'])
  config = config.update({
      'logdir': f'~/logdir/fetchslide_sparse',
      'run.steps': 1e8,
      'run.train_ratio': 64,
      'run.log_every': 300,  # Seconds
      'run.eval_every': 1e4,
      'run.eval_eps': 1,
      'batch_size': 16,
      'jax.prealloc': False,
      'encoder.mlp_keys': '$^',
      'decoder.mlp_keys': '$^',
      'encoder.cnn_keys': 'image|goal_image',
      'decoder.cnn_keys': 'image|goal_image',
      'wrapper.length': 1000,
      # 'jax.platform': 'cpu',
  })
  config = embodied.Flags(config).parse()

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      embodied.logger.WandBOutput(logdir, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])

  from embodied.envs.fetch import FetchSlide

  env = FetchSlide(sparse_reward=True)
  env = dreamerv3.wrap_env(env, config)
  env = embodied.BatchEnv([env], parallel=False)

  eval_env = FetchSlide(sparse_reward=True)
  eval_env = dreamerv3.wrap_env(eval_env, config)
  eval_env = embodied.BatchEnv([eval_env], parallel=False)

  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  replay = embodied.replay.Uniform(
      config.batch_length, config.replay_size, logdir / 'replay')
  eval_replay = embodied.replay.Uniform(
      config.batch_length, config.replay_size // 10, logdir / 'eval_replay')
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)

#   embodied.run.train(agent, env, replay, logger, args)
#   embodied.run.eval_only(agent, env, logger, args)
  embodied.run.train_eval(agent, env, eval_env, replay, eval_replay, logger, args)


if __name__ == '__main__':
  main()