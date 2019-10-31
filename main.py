from envs_layer import ArenaRllibEnv

trainer = pg.PGAgent(env="ArenaRllibEnv", config={
    "multiagent": {
        "policies": {
            # the first tuple value is None -> uses default policy
            "P0": (None, car_obs_space, car_act_space, {"gamma": 0.85}),
            "P1": (None, car_obs_space, car_act_space, {"gamma": 0.99}),
        },
    },
})

while True:
    print(trainer.train())
