from gym.envs.registration import register

register(id='SSLGoToBall-v0',
         entry_point='rsoccer_gym.ssl.ssl_go_to_ball:SSLGoToBallEnv',
         kwargs={'field_type': 2, 'n_robots_yellow': 6},
         max_episode_steps=1200
         )