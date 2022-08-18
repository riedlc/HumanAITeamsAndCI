from scripts.helper import *

# Get data
with open('../data/clues.json') as f:
    clues_dict = json.load(f)

with open('../data/messages.json') as f:
    chat_val_ordered = json.load(f)

info_labels = {-0.5: 'N', -0.25: 'MN', 0: None, 0.25: 'MY', 0.5: 'Y'}

culprit_clue_vals = [[0, 0.25, 0, 0, 0.25],
                     [-.5, 0, 0, 0, 0],
                     [0, -.5, 0, 0, 0],
                     [0.25, 0, 0.25, 0, 0],
                     [0, 0, 0.25, 0.25, 0],
                     [0, 0, 0, -0.5, -0.5]]

culprit_clue_labels = ['Maybe 2 or 5',
                       'Not 1',
                       'Not 2',
                       'Maybe 1 or 3',
                       'Maybe 3 or 4',
                       'Not 4 or 5']

message_labels = {-0.5: 'N', -0.25: 'MN', 0: None, 0.25: 'MY', 0.5: 'Y'}

# Get each team's adjacency matrix
adj_dict = get_adj_dict()

responses_dict = {}
for _, response in responses.iterrows():
    gid = response['GAMEID']
    if gid not in responses_dict.keys():
        responses_dict[gid] = [None] * 5
    responses_dict[gid][response['SubjectID'] - 1] = dict(response)


class SubAgentMemory:
    '''
    This is a mental model
    It collects information and can aggregate the learned information
    '''

    def __init__(self):
        self.exists = False
        self.dists = [0.2] * 5
        self.prior = [0.2] * 5
        self.info = []

    def update(self, line):
        self.exists = True
        self.info.append(line)

    def discern(self, surprise_weighting=False):
        '''
        Compute posteriors given the information observed
        :param surprise_weighting:
        :return:
        '''

        if self.info == []:
            return [0.2] * 5

        # enumerate through all information for each culprit
        # Translate the info matrix so we can iterate by culprit
        info_T = np.array(self.info).T

        for i, mentions in enumerate(info_T):
            for m in mentions:
                if surprise_weighting:
                    if m > 0:
                        weight = -np.log(self.dists[i])
                    elif m < 0:
                        weight = -np.log(1 - self.dists[i])
                    else:
                        continue

                    self.dists[i] *= self.conversion[m] ** weight
                else:
                    self.dists[i] *= self.conversion[m]

        return self.dists

    def tmp_discern(self, surprise_weighting=False):
        '''
        Computer hypothetical posteriors but do not change any variables
        :param surprise_weighting:
        :return:
        '''
        if self.info == []:
            return [0.2] * 5

        # enumerate through all information for each culprit
        # Translate the info matrix so we can iterate by culprit
        info_T = np.array(self.info).T

        dists = deepcopy(self.dists)
        for i, mentions in enumerate(info_T):
            for m in mentions:
                if surprise_weighting:
                    if m > 0:
                        weight = -np.log(dists[i])
                    elif m < 0:
                        weight = -np.log(1 - dists[i])
                    else:
                        continue

                    dists[i] *= self.conversion[m] ** weight
                else:
                    dists[i] *= self.conversion[m]

        return dists

    def set_conversion(self, conversion):
        self.conversion = conversion

    def get_prior(self):
        if self.info == []:
            return [0.2] * 5
        # enumerate through all information for each culprit
        info_T = np.array(self.info[:2]).T

        for i, mentions in enumerate(info_T):
            for m in mentions:
                self.prior[i] *= self.conversion[m]

        return self.prior


class Agent:
    '''
    This is a model of the player
    Agents have self.teammates, which is the agent's mental models for each teammate
    self.teammates[self.agent_id] is an agent's mental model of itself
    '''

    def __init__(self, agent_id, team_id, pub_clue, pri_clue, conversion, teammates, surprise_weighting,
                 tom, norm):
        self.agent_id = agent_id
        self.team_id = team_id
        self.teammates = teammates
        self.tom = tom
        self.pub_clue = pub_clue
        self.pri_clue = pri_clue
        self.norm = norm
        self.surprise_weighting = surprise_weighting

        for a in teammates:
            a.set_conversion(conversion)

        self.teammates[agent_id].update(pub_clue)
        self.teammates[agent_id].update(pri_clue)
        self.prior = self.norm(deepcopy(self.teammates[agent_id].get_prior()))

    def update(self, entity, line):
        self.teammates[entity].update(line[3:8])

    def discern(self):
        '''
        Infer a posterior based on the posteriors of an agent's mental models (teammates/SubAgents)
        :return:
        '''
        positions = []

        for t in self.teammates:
            if t.exists:
                positions.append(self.norm(t.discern(self.surprise_weighting)))
            else:
                positions.append(None)

        # To emphasize a particular distribution, we need to change the relative distances
        # between each of the 5 entries. Just multiplying does nothing.
        # Here I manipulate the distance from 1 by the self.tom parameter
        for i in range(len(self.teammates)):
            if i == self.agent_id or positions[i] is None:
                continue
            positions[i] = [x ** self.tom for x in positions[i]]

        result = np.array([1] * 5)
        for p in positions:
            if p is not None:
                result = result * p
        return self.norm(result)

    def tmp_discern(self, random_info=False):
        '''
        Test hypothetical posteriors given specific interventions
        :param random_info: If true - conduct a random intervention
        :return: a message to send that lowers the DKL between the ago and alter parameters
        '''
        info = deepcopy(self.teammates[self.agent_id].info)
        if random_info:
            info.append(None)
        mess = [None] * 5
        self_position = self.norm(self.teammates[self.agent_id].tmp_discern(self.surprise_weighting))
        for player_id, t in enumerate(self.teammates):
            if random_info:
                choice = np.random.randint(len(info))
                mess[player_id] = info[choice]
                continue
            if t.exists and player_id != self.agent_id:
                null_position = self.norm(t.tmp_discern(self.surprise_weighting))
                proto_positions = [None] * len(info)
                for i, tmp_info in enumerate(info):
                    t.update(tmp_info)
                    proto_positions[i] = self.norm(t.tmp_discern(self.surprise_weighting))
                    t.info = t.info[:-1]

                min_dkl = KL(self_position, null_position)
                for i, proto in enumerate(proto_positions):
                    if proto is not None:
                        dkl = KL(self_position, proto)
                        if dkl < min_dkl:
                            min_dkl = dkl
                            mess[player_id] = info[i]

                # same answer

        return mess


def tom_simulate(conversion, sub_agent, tom, ego_loop, return_prior, norm,
                 surprise_weighting):
    '''
    Simulate the model for every team and player
    :param conversion: Information weights. Of the form
            {-0.5: sn, -0.25: mn, 0: 1, 0.25: my, 0.5: sy}
    :param sub_agent: Type of mental model. This is always SubAgentMemory
    :param tom: Theory of Mind parameter Alpha_D
    :param ego_loop: If true - include the self-actualization loop in the ego models
    :param return_prior: If true - return a second set of distributions that is the prior for each agent
    :param norm: Method to normalize values
    :param surprise_weighting: If true - weight incoming information by it's surprise
    :return: Posterior distributions for each player
    '''


    # Simulate
    tom_dict = {}
    prior_dict = {}
    for team_id, val in chat_val_ordered.items():
        clues = clues_dict[team_id]

        pub_clue = culprit_clue_vals[clues['public'][0] - 1]

        team = []
        for x in range(5):
            team.append(Agent(x, team_id, pub_clue, culprit_clue_vals[clues[str(x + 1)][0] - 1], conversion,
                            [sub_agent() for _ in range(5)], tom=tom, norm=norm,
                            surprise_weighting=surprise_weighting))

        # Chat data
        for line in val:
            speaker = line[0] - 1
            listeners = deepcopy(line[1])
            if ego_loop:
                listeners.append(line[0])
            for listener in listeners:
                listener -= 1
                team[listener].update(speaker, line)

        tom_dict[team_id] = [agent.discern() for agent in team]
        prior_dict[team_id] = [agent.prior for agent in team]

    if return_prior:
        return tom_dict, prior_dict
    return tom_dict


def tom_simulate_message_intervention(conversion, sub_agent, tom, ego_loop, return_prior=False, norm=normalize,
                                      surprise_weighting=False, random_info=False):
    '''
    The same as the tom_simulate method above, but also perform the final intervention stpe
    :param conversion:
    :param sub_agent:
    :param tom:
    :param ego_loop:
    :param return_prior:
    :param norm:
    :param surprise_weighting:
    :param random_info: if true - perform a random intervention
    :return:
    '''
    tom_dict = {}
    prior_dict = {}
    for team_id, val in chat_val_ordered.items():
        clues = clues_dict[team_id]

        pub_clue = culprit_clue_vals[clues['public'][0] - 1]

        team = [None] * 5
        for x in range(5):
            team[x] = Agent(x, team_id, pub_clue, culprit_clue_vals[clues[str( x +1)][0] - 1], conversion,
                            [sub_agent() for _ in range(5)], tom=tom, norm=norm,
                            surprise_weighting=surprise_weighting)

        # Chat data
        for line in val:
            speaker = line[0] - 1
            listeners = deepcopy(line[1])
            if ego_loop:
                listeners.append(line[0])
            for listener in listeners:
                listener -= 1
                team[listener].update(speaker, line)

        for x in range(len(team)):
            new_lines = team[x].tmp_discern(random_info=random_info)
            for listener in range(len(new_lines)):
                if new_lines[listener] is not None:
                    team[listener].teammates[x].update(new_lines[listener])

        tom_dict[team_id] = [agent.discern() for agent in team]
        prior_dict[team_id] = [agent.prior for agent in team]

    if return_prior:
        return tom_dict, prior_dict
    return tom_dict


def random_agent():
    '''
    Equivalent to an agent that randomly responds to this task
    :return:
    '''
    count = 0
    match = 0
    total = 0
    # responses = pd.read_csv('../data/responses.csv')
    for _, row in responses.iterrows():
        guess = np.random.randint(5)
        if guess == 2:
            count += 1
        if guess == row['Culprit']-1:
            match += 1
        total += 1
    return count, match, total