from scripts.simulation import *


def generate_results():
    # responses = pd.read_csv('../data/responses.csv')

    # prior results
    prior_performance = []
    prior_agg = []
    np.random.seed(100)
    conversion = {-0.5: 0.1, -0.25: 0.2, 0: 1, 0.25: 1.8, 0.5: 2}
    for i in range(100):
        tom_a, tom_no_a = tom_simulate(conversion, SubAgentMemory, tom=1, return_prior=True, norm=normalize,
                                       ego_loop=True, surprise_weighting=True)
        prior_performance.append(performance(tom_no_a)[0]/145*100)
        prior_agg.append(human_agent_match(tom_no_a)[0]/145*100)

    print(f'Prior        \t performance: \t {round(np.mean(prior_performance), 1)} pm {round(np.std(prior_performance), 1)} \t human-agent agreement: \t '
          f'{round(np.mean(prior_agg), 1)} pm {round(np.std(prior_agg), 1)}'
    )


    # self-actualization
    conversion = {-0.5: 0.05, -0.25: 0.05, 0: 1, 0.25: 1.5, 0.5: 2}
    p = []
    m = []
    for _ in range(100):
        tom_a, tom_no_a = tom_simulate(conversion, SubAgentMemory, tom=0, ego_loop=True, return_prior=True,
                                       norm=normalize, surprise_weighting=True)
        p.append(performance(tom_a)[0]/145*100)
        m.append(human_agent_match(tom_a)[0]/145*100)
    print(f'Self-act.     \t performance: \t {round(np.mean(p), 1)} pm {round(np.std(p), 1)} \t human-agent agreement: \t '
          f'{round(np.mean(m), 1)} pm {round(np.std(m), 1)}'
    )

    # partner actualization
    conversion = {-0.5: 0.15, -0.25: 1, 0: 1, 0.25: 1.55, 0.5: 2}
    p = []
    m = []
    for _ in range(100):
        tom_a, tom_no_a = tom_simulate(conversion, SubAgentMemory, tom=1, ego_loop=False, return_prior=True,
                                       norm=normalize, surprise_weighting=True)
        p.append(performance(tom_a)[0]/145*100)
        m.append(human_agent_match(tom_a)[0]/145*100)
    print(f'Partner-act. \t performance: \t {round(np.mean(p), 1)} pm {round(np.std(p), 1)} \t human-agent agreement: \t '
          f'{round(np.mean(m), 1)} pm {round(np.std(m), 1)}'
    )


    # MLE
    conversion = {-0.5: 0.1, -0.25: 1, 0: 1, 0.25: 1.45, 0.5: 2}
    p = []
    m = []
    for _ in range(100):
        tom_a, tom_no_a = tom_simulate(conversion, SubAgentMemory, tom=0.95, ego_loop=True, return_prior=True,
                                       norm=normalize, surprise_weighting=True)
        p.append(performance(tom_a)[0]/145*100)
        m.append(human_agent_match(tom_a)[0]/145*100)
    print(
        f'MLE         \t performance: \t {round(np.mean(p), 1)} pm {round(np.std(p), 1)} \t human-agent agreement: \t '
        f'{round(np.mean(m), 1)} pm {round(np.std(m), 1)}'
    )

    # Max accuracy
    conversion = {-0.5: 0.35, -0.25: .85, 0: 1, 0.25: 1.95, 0.5: 2}
    p = []
    m = []
    for _ in range(100):
        tom_a, tom_no_a = tom_simulate(conversion, SubAgentMemory, tom=0.95, ego_loop=True, return_prior=True,
                                       norm=normalize, surprise_weighting=True)
        p.append(performance(tom_a)[0]/145*100)
        m.append(human_agent_match(tom_a)[0]/145*100)
    print(
        f'Max perf.      \t performance: \t {round(np.mean(p), 1)} pm {round(np.std(p), 1)} \t human-agent agreement: \t '
        f'{round(np.mean(m), 1)} pm {round(np.std(m), 1)}'
    )

    for _ in range(200):
        np.random.randint(2)

    # Max agreement
    conversion = {-0.5: 0.05, -0.25: .75, 0: 1, 0.25: 1.25, 0.5: 1.95}
    p = []
    m = []
    for _ in range(100):
        tom_a, tom_no_a = tom_simulate(conversion, SubAgentMemory, tom=0.45, ego_loop=True, return_prior=True,
                                       norm=normalize, surprise_weighting=True)
        p.append(performance(tom_a)[0]/145 * 100)
        m.append(human_agent_match(tom_a)[0]/145 * 100)
    print(
        f'Max agree      \t performance: \t {round(np.mean(p), 1)} pm {round(np.std(p), 1)} \t human-agent agreement: \t '
        f'{round(np.mean(m), 1)} pm {round(np.std(m), 1)}'
    )

    # Random results
    m = []
    p = []
    for _ in range(100):
        out = random_agent()
        m.append(out[1] / 145 * 100)
        p.append(out[0] / 145 * 100)

    print(
        f'Random       \t performance: \t {round(np.mean(p), 1)} pm {round(np.std(p), 1)} \t human-agent agreement: \t '
        f'{round(np.mean(m), 1)} pm {round(np.std(m), 1)}'
    )

    # Intervention
    conversion = {-0.5: 0.35, -0.25: .85, 0: 1, 0.25: 1.95, 0.5: 2}
    np.random.seed(102)
    rand_improvements = []
    rand_hum = []
    improvements = []
    impr_hum = []

    for i in range(100):
        tom_a_rand = tom_simulate_message_intervention(conversion, SubAgentMemory, tom=.95, return_prior=False,
                                                       norm=normalize,
                                                       ego_loop=True, surprise_weighting=True, random_info=True)
        tom_a = tom_simulate_message_intervention(conversion, SubAgentMemory, tom=.95, return_prior=False,
                                                  norm=normalize,
                                                  ego_loop=True, surprise_weighting=True, random_info=False)
        rand_improvements.append(performance(tom_a_rand)[0])
        rand_hum.append(human_agent_match(tom_a_rand)[0])
        improvements.append(performance(tom_a)[0])
        impr_hum.append(human_agent_match(tom_a)[0])

    ri = [x / 145 * 100 for x in rand_improvements]
    rh = [x / 145 * 100 for x in rand_hum]
    imp = [x / 145 * 100 for x in improvements]
    himp = [x / 145 * 100 for x in impr_hum]
    print(
        f'Rand. inter.   \t performance: \t {round(np.mean(ri), 1)} pm {round(np.std(ri), 1)} \t human-agent agreement: \t '
        f'{round(np.mean(rh), 1)} pm {round(np.std(rh), 1)}'
    )
    print(
        f'Intervention\t performance: \t {round(np.mean(imp), 1)} pm {round(np.std(imp), 1)} \t human-agent agreement: \t '
        f'{round(np.mean(himp), 1)} pm {round(np.std(himp), 1)}'
    )

    print('Test statistic between random vs. targeted intervention:')
    tt = ttest_ind(ri, imp)
    print(f'Stat: {round(tt[0], 8)}, p-val: {round(tt[1]), 8}')




if __name__ == '__main__':
    generate_results()
