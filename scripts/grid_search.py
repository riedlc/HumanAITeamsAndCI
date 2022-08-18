# Multithreading resource: https://www.geeksforgeeks.org/multithreading-python-set-1/

import threading
from scripts.simulation import *
import os


def get_results(grid_all, conversion, tom, norm, ego_loop):  # , no_tom_results, prior_dict):
	tom_results, tom_prior = tom_simulate(conversion, SubAgentMemory, tom=tom, return_prior=True, norm=norm,
	                                      ego_loop=ego_loop, surprise_weighting=True)
	# wb_agent = Weak_Bayes(conversion, alpha=tom, surprise_weighting=True)
	# no_tom_results = wb_agent.simulate(norm=norm, ego_loop=ego_loop)

	for game in tom_results.keys():
		degs = [sum(x) for x in adj_dict[str(game)]]
		for player in range(5):
			pub_clue_idx = clues_dict[game]['public'][0] - 1
			pri_clue_idx = clues_dict[game][str(player + 1)][0] - 1
			pub_clue = culprit_clue_vals[pub_clue_idx]
			pri_clue = culprit_clue_vals[pri_clue_idx]
			prior = tom_prior[game]

			hum_ans = responses_dict[int(game)][player]['Culprit'] - 1
			h_correct = 1 if hum_ans == 2 else 0

			tom_post = tom_results[game][player]
			#             tom_ans = np.argmax(tom_post)
			top = np.argwhere(tom_post == np.amax(tom_post))
			tom_ans = top[np.random.randint(len(top))][0]
			tom_correct = 1 if tom_ans == 2 else 0
			tom_h_match = 1 if tom_ans == hum_ans else 0

			# no_tom_post = no_tom_results[game][player]
			#             no_tom_ans = np.argmax(no_tom_post)
			# top = np.argwhere(no_tom_post == np.amax(no_tom_post))
			# no_tom_ans = top[np.random.randint(len(top))][0]
			# no_tom_correct = 1 if no_tom_ans == 2 else 0
			# no_tom_h_match = 1 if no_tom_ans == hum_ans else 0

			# tom_no_tom_match = 1 if tom_ans == no_tom_ans else 0

			line = {'TeamID': game, 'SubjectID': player, 'Degree': degs[player], 'Alpha': tom, 'HumanAns': hum_ans,
			        'HumCor': h_correct, 'TomAns': tom_ans, 'TomCor': tom_correct,
			        # 'NoTomAns': no_tom_ans, 'NoTomCor': no_tom_correct,
			        'HumTomMatch': tom_h_match, 'HumNoTomMatch': None,  # no_tom_h_match,
			        # 'TomNoTomMatch': tom_no_tom_match,
			        'PubClueIdx': pub_clue_idx, 'PriClueIdx': pri_clue_idx}
			for i in range(5):
				x = pub_clue[i]
				if x == 0:
					x = None
				line['PubClue' + str(i)] = x
			for i in range(5):
				x = pri_clue[i]
				if x == 0:
					x = None
				line['PriClue' + str(i)] = x
			for i in range(5):
				line['Prior' + str(i)] = np.around(prior[player][i], 6)
			for i in range(5):
				line['TomPost' + str(i)] = np.around(tom_post[i], 6)
			#             for i in range(5):
			#                 line['NoTomPost'+str(i)] = no_tom_post[i]
			for key, val in conversion.items():
				if key == 0:
					continue
				line[info_labels[key]] = val
			grid_all.append(line)


def search_params(lower, upper, ego_loop, save_path):
	count = 0
	grid_all = []
	for n in range(lower, upper):
		for mn in range(n, 21):
			for my in range(20, 41):
				for y in range(my, 41):
					conv = {-0.5: n, -0.25: mn, 0: 1, 0.25: my, 0.5: y}
					conv = {k: np.around(0.05 * v, 2) for k, v in conv.items()}
					conv[0] = 1
					for alpha in range(21):
						alpha = np.around(0.05 * alpha, 2)
						get_results(grid_all, conv, tom=alpha, norm=normalize, ego_loop=ego_loop)
						count += 1
			t_path = save_path + str(n) + '_' + str(mn) + '.csv'
			pd.DataFrame(grid_all).to_csv(t_path)
			grid_all = []


if __name__ =="__main__":
	self_actualize = True
	if not os.path.isdir('../data/grid_search'):
		os.mkdir('../data/grid_search')
	if not os.path.isdir('../data/grid_search/egoLoop'+str(self_actualize)):
		os.mkdir('../data/grid_search/egoLoop'+str(self_actualize))
	save_path = '../data/grid_search/egoLoop'+str(self_actualize)
	threads = []

	for i in range(1, 3): # i must be greater than 0
		threads.append(threading.Thread(target=search_params, args=(i, i+1, self_actualize, save_path)))
	for i in range(len(threads)):
		threads[i].start()
	for i in range(len(threads)):
		threads[i].join()
