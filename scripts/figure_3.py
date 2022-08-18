from scripts.simulation import *

def fig_3a(tom_a):
	# generate these stats so the random seed
	# matches plots from the paper
	_ = performance(tom_a), human_agent_match(tom_a), performance(tom_a), human_agent_match(tom_a)

	prior_list = {}
	for i in range(6):
		for j in range(6):
			if i == j:
				continue
			private_clue = culprit_clue_vals[i]
			public_clue = culprit_clue_vals[j]
			agt = SubAgentMemory()
			agt.set_conversion(conversion)
			agt.update(private_clue)
			agt.update(public_clue)
			prior_list[str(i) + str(j)] = np.around(normalize(agt.discern())[2], 6)

	hum_given_prior = {x: [] for x in prior_list.values()}
	tom_given_prior = {x: [] for x in prior_list.values()}
	tom_match_given_prior = {x: [] for x in prior_list.values()}

	for i, row in results.iterrows():
		game = str(row['GAMEID'])
		player = row['SubjectID'] - 1
		private_clue = clues_dict[game][str(row['SubjectID'])][0] - 1
		public_clue = clues_dict[game]['public'][0] - 1
		p = prior_list[str(private_clue) + str(public_clue)]
		hum_given_prior[p].append(row['Culprit'])

		selection = randmax(tom_a[game][player])
		c = 0
		if selection == 2:
			c = 1
		tom_given_prior[p].append(c)

		# Used to match random seed to paper figure
		_ = randmax(tom_a[game][player])

		selection = randmax(tom_a[game][player])
		c = 0
		if selection == responses.iloc[i]['Culprit'] - 1:
			c = 1
		tom_match_given_prior[p].append(c)

	hum_given_prior = dict(sorted(hum_given_prior.items(), key=lambda item: item[0]))
	tom_given_prior = dict(sorted(tom_given_prior.items(), key=lambda item: item[0]))
	tom_match_given_prior = dict(sorted(tom_match_given_prior.items(), key=lambda item: item[0]))

	shift = 0.001
	plt.plot([1 - x for x in tom_match_given_prior.keys()],
			 [np.mean(x) * 100 for x in tom_match_given_prior.values()],
			 marker='^', label='Human Agent Agreement', alpha=1, c=colors[0])
	plt.plot([1 - (x - shift) for x in hum_given_prior.keys()],
			 [np.mean(x) * 100 for x in hum_given_prior.values()],
			 marker='s', label='Human', alpha=1, c=colors[1])
	plt.plot([1 - (x + shift) for x in tom_given_prior.keys()],
			 [np.mean(x) * 100 for x in tom_given_prior.values()],
			 marker='o', label='ToM Agent', alpha=1, c=colors[2])

	plt.xlabel('Task Difficulty', fontsize=15)
	plt.ylabel('Average Performance', fontsize=15)
	plt.legend()
	plt.ylim(0, )
	ax = plt.gca()
	for spine in ['top', 'right']:
		ax.spines[spine].set_visible(False)
	labels = ax.get_yticks().tolist()
	labels = [str(int(x)) + "%" for x in labels]
	ax.set_yticklabels(labels)

	plt.tight_layout()
	plt.show()


def compare_degree(d):
	degrees = []
	total_correct = []
	culprit_correct = []
	human_match = []
	agent_correct = []
	for game in responses['GAMEID'].unique():
		deg_list = [sum(x) for x in adj_dict[str(game)]]
		degrees.extend(deg_list)
		team = results[results['GAMEID'] == game]
		for player_id in range(5):
			player = team[team['SubjectID'] == player_id + 1].iloc[0]
			total_correct.append(player['Total_Correct'])
			culprit_correct.append(player['Culprit'])
			model_guess = np.argmax(d[str(game)][player_id])
			human_guess = responses['Culprit'][(responses['GAMEID'] == game) &
											   (responses['SubjectID'] == player_id + 1)].iloc[0]
			human_match.append(1 if model_guess == (human_guess - 1) else 0)
			agent_correct.append(1 if model_guess == 2 else 0)

	return {'degrees': degrees, 'total_correct': total_correct,
			'culprit_correct': culprit_correct, 'human_match': human_match,
			'agent_correct': agent_correct}


def degree_bins(output):
	r_dict = compare_degree(output)
	total_correct_bin = [[] for _ in range(4)]
	culprit_correct_bin = [[] for _ in range(4)]
	human_match_bin = [[] for _ in range(4)]
	agent_correct_bin = [[] for _ in range(4)]
	for i in range(len(r_dict['degrees'])):
		deg = int(r_dict['degrees'][i])
		if deg == 0:
			# print('0 degree at index:', i)
			continue
		deg -= 1
		total_correct_bin[deg].append(r_dict['total_correct'][i])
		culprit_correct_bin[deg].append(r_dict['culprit_correct'][i])
		human_match_bin[deg].append(r_dict['human_match'][i])
		agent_correct_bin[deg].append(r_dict['agent_correct'][i])

	return total_correct_bin, culprit_correct_bin, human_match_bin, agent_correct_bin


def fig_3b(tom_a):
	total_correct_bin, culprit_correct_bin, human_match_bin, agent_correct_bin = degree_bins(tom_a)

	plt.plot([1, 2, 3, 4], [np.mean(x) * 100 - 0.5 for x in human_match_bin], marker='^', c=colors[0],
			 label='Human Agent Agreement', alpha=1)
	plt.plot([1, 2, 3, 4], [np.mean(x) * 100 + 0.5 for x in culprit_correct_bin], marker='s', c=colors[1],
			 label='Human Accuracy', alpha=1)
	plt.plot([1, 2, 3, 4], [np.mean(x) * 100 for x in agent_correct_bin], marker='o', c=colors[2],
			 label='ToM Agent Accuracy', alpha=1)
	plt.ylabel('Average Performance', fontsize=15)
	plt.xlabel('Communication Partners', fontsize=15)
	plt.xticks([1, 2, 3, 4])
	plt.ylim(0, 100)

	ax = plt.gca()
	for spine in ['top', 'right']:
		ax.spines[spine].set_visible(False)

	labels = ax.get_yticks().tolist()
	labels = [str(int(x)) + "%" for x in labels]
	ax.set_yticklabels(labels)

	plt.legend(loc='lower right')
	plt.tight_layout()
	plt.show()


def fig_3c(tom_a):

	culprit_by_alpha = []
	match_by_alpha = []
	agent_by_alpha = []


	vals = [x / 100 for x in range(101)]
	tmp = None
	for alpha in vals:
		tom_a, tom_no_a = tom_simulate(conversion, SubAgentMemory, tom=alpha, return_prior=True, norm=softmax,
									   ego_loop=False, surprise_weighting=True)
		binned = degree_bins(tom_a)
		culprit_by_alpha.append([np.mean(x) for x in binned[1]])
		match_by_alpha.append([np.mean(x) for x in binned[2]])
		agent_by_alpha.append([np.mean(x) for x in binned[3]])


	fig, ax = plt.subplots()
	matches = np.array(match_by_alpha).T
	agents = np.array(agent_by_alpha).T
	for i in range(4):
		ax.plot(vals, [int(x * 100) for x in agents[i]], label='ToM Agents with ' + str(i + 1) + ' Neighbors',
				c=colors[i])

	labels = ax.get_yticks().tolist()
	labels = [str(int(x)) + "%" for x in labels]
	ax.set_yticklabels(labels)

	for spine in ['top', 'right']:
		ax.spines[spine].set_visible(False)

	ax.legend(loc='lower right')
	ax.set_xlabel(r'Theory of Mind Ability $\alpha$', fontsize=15)
	ax.set_ylabel('Average Performance', fontsize=15)
	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	np.random.seed(101)
	plt.rcParams.update({'font.size': 13})
	conversion = {-0.5: 0.35, -0.25: .85, 0: 1, 0.25: 1.95, 0.5: 2}

	tom_a, tom_no_a = tom_simulate(conversion, SubAgentMemory, tom=0.95, return_prior=True, norm=normalize,
								   ego_loop=True, surprise_weighting=True)
	fig_3a(tom_a)
	fig_3b(tom_a)
	fig_3c(tom_a)
