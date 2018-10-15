"""
It have to tell trader to bay or not
Neuron network MoneyMaker
"""
import math, random, numpy, sys, time, os
from numpy import median

# Hyper parametrs start
we_need_neurons = ['S',3],['A',2],['R',1] #setup
active_neurons = ([0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]) # learning batch
correct_result = [0,1,0,0,1,1,0,1] # expected result
learnin_rate = 0.008 # how fast will network learning (jumping)
variable_learning_rate = True
epoch = 3000 # times of learning
times_of_network_intialize = 10
clever_random = True # initialize clever random weights if sinapses
stable_random_seed = False # generate same random every time
# hyper parametrs end

best_result = 1.0 # counter for best sicle
total_errors = 0 # counter of errors in learning batch
total_errors_for_new_questions= 0 # counter of errors in new batch

new_questions = ([0.01,0.1,0.05],[0.2,0,1.03],[0.08,1.2,0.2],[0,0.9,0.85],[1.11,0.003,0],[1,0,1.2],[0.79,1,0],[1,1,1.002])
correct_answers_for_new_questions = [0,1,0,0,1,1,0,1]
total_errors_for_new_questions= 0

neurons = [] # create list of neurons
layers_names = [] # we need to decide levels, witch could be different
for setup_layer in we_need_neurons:
	globals()[setup_layer[0]] = [] # now, we create A,S,R,etc lists
W = [] # list of sinapses

class Neuron:
	id = int
	fun = str
	activation_function_result = 0.0 # we need it to learning
	neuron_delta = 0.0
	neuron_error = 0

	def __init__(self): # add new neuron to neurons list
	   self.id = len(neurons) + 1 # create neuron ID
	   neurons.append(self)

	def activation_function(self):
		global W
		limit = 0.5
		if self.fun != we_need_neurons[0][0]: # not for sensor layer
			tW = 0.0
			for i in W:
				if i.connection_to == self:
					tW += i.connection_from.activation_function_result * i.weight
			sigmoid = 1/(1+numpy.exp(-tW))
			self.activation_function_result = sigmoid # save to neuron var
			if sigmoid >= limit: 
				return True
			else:
				return False
		else:
			if self.activation_function_result >= limit: return True 
			else: return False

	def find_neuron_error(self, stack_number):
		global W, correct_result
		if self.fun != 'R' and self.fun != 'S':
			for w in W:
				if w.connection_from == self:
					self.neuron_error += w.connection_to.neuron_error * w.weight
					# print(self.neuron_error)
		elif self.fun == 'R':
			self.neuron_error = correct_result[stack_number] - self.activation_function_result

	def derivative_of_sigmoid_fun(self):
		derivative_of_sigmoid_fun_result = self.activation_function_result * (1 - self.activation_function_result)
		self.neuron_delta = self.neuron_error * derivative_of_sigmoid_fun_result

class Sinapse:
	id = int
	connection_from = None
	connection_to = None
	weight = 0.0

	def __init__(self): # add new neuron to neurons list
	   self.id = len(W) + 1 # create neuron ID
	   W.append(self)
	
	def random_weight(self):
		global S, R, clever_random, stable_random_seed
		if stable_random_seed: # for control
			numpy.random.seed(1) 
		else: # to generate absolutely random random
			numpy.random.seed()
		if clever_random == False:
			n = random.random()
		else:
			x = 4 * math.sqrt(6/(len(S)+len(R)))
			n = random.triangular(-x, x)
		self.weight = n


# now, let's create neurons, append them to lists and set them function
for n in we_need_neurons: # for every level
	for i in range (1, n[1] + 1): # becouse we start from one
		layer_name = n[0] 
		globals()[layer_name+'%s' % i] = Neuron() # create new neuron object
		globals()[layer_name].append(globals()[layer_name+'%s' % i]) # append to list
		globals()[layer_name][-1].fun = layer_name # set function of last element of neurons list
			
# lets activete every neuron we need to be activated at start by default
def activate_sensor_neurons(stack_number):
	global active_neurons, S
	stack_for_study = active_neurons[stack_number] 
	for i in range(len(stack_for_study)):
		for n in S:
			if n.id == i+1:
				n.activation_function_result = stack_for_study[i]
				print(stack_for_study, n.id, stack_for_study[i])

# now, we count how mutch sinapses does we need and create them
# we_need_sinapses = len(S) * len (A) + len (A) * len (R) # count how many sin we need
we_need_sinapses = 0
layer_n_count = [] # we need to decide levels, witch could be different
for setup_layer in we_need_neurons:
	layer_n_count.append(setup_layer[1])
for i in range(len(layer_n_count)-1): # level per level
	we_need_sinapses += layer_n_count[i] * layer_n_count[i+1]
for i in range (1, we_need_sinapses + 1): # becouse we start from one
	globals()['W%s' % i] = Sinapse() # create new sinapse object

def connect(): # connect every sinapses for every level
	global W, neurons
	used_connections = [] # to do not create the same connections
	layers_names = [] # we need to decide levels, witch could be different
	for setup_layer in we_need_neurons:
		layers_names.append(setup_layer[0])
	for i in range(len(layer_n_count)-1):
		neurons_from = globals()[layers_names[i]]
		neurons_to = globals()[layers_names[i+1]]
		for n_f in neurons_from:
			for n_t in neurons_to:
				for w in W:
					if w.connection_from is None and (str(n_f.id) + " " + str(n_t.id)) not in used_connections:
						w.connection_from = n_f
						w.connection_to = n_t
						used_connections.append(str(n_f.id) + " " + str(n_t.id))
						w.random_weight() # set random weight

connect()

def spyke(): # activate or deactivate every neuron layer by layer
	global neurons, R1
	layer_names = [] # we need to decide levels, witch could be different
	for setup_layer in we_need_neurons:
		layer_names.append(setup_layer[0])
	for level in layer_names: # level per level
		# if level != 'S':
		for n in neurons:
			if n.fun == level:
				n.activation_function()
	return R1.activation_function()

def sort_by_id(obj): # not so easy to sort arreys of objects
	return obj.id

def learn(): # find eror, and correct weights using back propogation mehtod
	global neurons, W, R, R1, correct_result, learnin_rate, active_neurons
	results = []
	for stack_number in range(len(active_neurons)):
		activate_sensor_neurons(stack_number)
		spyke()
		
		neurons.sort(key = sort_by_id, reverse = True) # was 1-2-3, now 3-2-1
		for n in neurons:
			n.neuron_error = 0 # reset
			n.find_neuron_error(stack_number)
			n.derivative_of_sigmoid_fun()
		
		for w in W:
			weight = w.weight + w.connection_to.neuron_delta * w.connection_from.activation_function_result * learnin_rate
			w.weight = weight

		cr = correct_result[stack_number]
		error = cr - R1.activation_function_result
		result = numpy.mean(error**2)
		results.append(result)
	median_result = median(results)
	return median_result

def print_network():
	print('\rNetwork state:')
	for w in W:
		print('S ID{0:3}: {1:1}{2:3} - {3:1}{4:3}, W = {5:3}'.format(w.id, w.connection_from.fun, w.connection_from.id, w.connection_to.fun, w.connection_to.id, w.weight))
	for n in neurons:
		print('{0:1}{1:3} result: {2:3}'.format(n.fun, n.id, n.activation_function_result))
# activate_sensor_neurons(1)
# spyke()        
# print_network()
def print_process (sycle, result, n_q_result, start_time):
    # time.sleep(0.01)
    global epoch
    percent = float(sycle) / (float(epoch) / 100)
    sys.stdout.write('\rError: {0:.3f} ({4:.3f} on new data). Epoch {1:4d} ({2:3}%) {3:3} sec'.format(round(result,4), sycle, round(percent,1), int(time.time() - start_time), round(n_q_result,2)))
    # sys.stdout.write("\rError: %s" % round(result,4) + ', epoch ' + str(sycle) + ' (' + str(round(percent,1)) + '%)')
    sys.stdout.flush()

def learn_every_epoch(): # learning process throw all epoch
	global correct_result, epoch, learnin_rate, best_result, variable_learning_rate
	learnin_rate_constant = learnin_rate
	devations = [] # network results
	start_time = time.time()
	for i in range(1,epoch+1):
		result = learn() # errors rate after learning
		n_q_result = new_questions_control(True) # errors on new questions
		print_process(i, result, n_q_result, start_time)
		# if variable_learning_rate == True:
		# 	if i == 1:
		# 		learnin_rate = learnin_rate_constant / 4
		# 	if i == 5:
		# 		learnin_rate = learnin_rate_constant * 2
		# 	if result < 0.25:
		# 		learnin_rate = learnin_rate_constant / 4
		# 	if result < 0.1:
		# 		learnin_rate = learnin_rate_constant / 6
		# 	if result < 0.05:
		# 		learnin_rate = learnin_rate_constant / 10
			# initial_lrate = 0.2
			# k = 0.01
			# learnin_rate = initial_lrate * numpy.exp(-k*i)
			 
		if result < 0.09 and n_q_result < 0.2:
			print('say "Learning has been stoped at epoch ' + str(i) + ' couse it good"')
			break
		devations.append(result) # add
		if times_of_network_intialize > 1:
			if n_q_result + result < best_result and n_q_result + result < 0.2:
				best_result = n_q_result + result
				save_network(str(best_result))
	print('\rDeviation history:')
	i = 0
	for r in devations:
		i += 1
		if i % 100 == 0 or i == 1: # every 100 epoch
			print('   Epoch ' + str(i) + ': ' + str(round(r,4)))

def control_question(silent=False):
	global active_neurons, correct_result, total_errors, R1
	for stack_number in range(len(active_neurons)):
		activate_sensor_neurons(stack_number)
		spyke()
		active_n = active_neurons[stack_number]
		correct_res = correct_result[stack_number]
		prediction = R1.activation_function_result
		prediction_result = round(R1.activation_function_result)
		if prediction_result == correct_res: 
			result = 'CORRECT'
		else: 
			result = 'ERROR'
			total_errors += 1
		if silent is not True:
			print('### CONTROL: for S = ' + str(active_n) + ' correct res = ' + str(correct_res) + '. Prediction is ' + str(round(prediction,4)) + '. Prediction result = ' + str(prediction_result) + ' | ' + result)
	if silent is not True:
		print_network()
		if total_errors == 0:
			print('RESULT: NO ERRORS AFTER LEARNING')
		else:
			print('RESULT: HERE IS ' + str(total_errors) + ' of ' + str(len(active_neurons)) + ' ERRORS AFTER LEARNING')

def new_questions_control(silent=False):
	global new_questions, correct_answers_for_new_questions, total_errors_for_new_questions, R1
	n_q_results = []
	total_errors_for_new_questions = 0
	for stack_number in range(len(new_questions)):
		activate_sensor_neurons(stack_number)
		spyke()
		active_n = new_questions[stack_number]
		correct_res = correct_answers_for_new_questions[stack_number]
		prediction = R1.activation_function_result
		prediction_result = round(R1.activation_function_result)
		if prediction_result == correct_res: 
			result = 'CORRECT'
		else: 
			result = 'ERROR'
			total_errors_for_new_questions += 1
		if silent is not True:
			print('### NEW QUEST: for S = ' + str(active_n) + ' correct res = ' + str(correct_res) + '. Prediction is ' + str(round(prediction,4)) + '. Prediction result = ' + str(prediction_result) + ' | ' + result)
		
		error = correct_res - R1.activation_function_result
		result = numpy.mean((error-correct_res)**2)
		n_q_results.append(result)
	median_result = median(n_q_results)
	# print_network()
	if silent is not True:
		if total_errors_for_new_questions == 0:
			print('RESULT: NO ERRORS IN NEW QUEST')
		else:
			print('RESULT: HERE IS ' + str(total_errors_for_new_questions) + ' of ' + str(len(new_questions)) + ' ERRORS IN NEW QUEST')
	return median_result

def save_network(network_name):
	global total_errors, active_neurons, total_errors_for_new_questions, new_questions, we_need_neurons, learnin_rate, epoch, neurons, W
	f = open(str(network_name) + '.txt', 'w') # create new file if none
	f.write('# Neuron network: ' + network_name + '\n' + '# Total errors per learning: ' + str(total_errors) + ' / ' + str(len(active_neurons)) + ' . Total errors on new data: ' + str(total_errors_for_new_questions) + ' / ' + str(len(new_questions)) + '\n' + '# Architecture: ' + str(we_need_neurons) + '\n# Trained on default learning rate ' + str(learnin_rate) + ' through ' + str(epoch) + ' epoch.' + '\n')
	for n in neurons:
		f.write('N/' + str(n.fun) + '/' + str(n.id) + '/' + str(n.activation_function_result) + '\n')
	for w in W:
		f.write('S/' + str(w.id) + '/' + str(w.connection_from.id) + '/' + str(w.connection_to.id) + '/' + str(w.weight) + '\n')
	f.close()


print('What you want to do?\n 1 - Create new network and tech it by default settings \n 2 - Load network from file \n 3 - Create new network many times and find best one')
new_or_load_input = input()

if new_or_load_input == '1': # Create new network and tech it by default settings
	learn_every_epoch()
	control_question()
	new_questions_control()
	os.system('say "The neural network is trained!"')
	print('Do you want to save network state?\n y - Yes \n n - No')
	save_input = input()
	if save_input == 'y':
		print('Enter name of network:')
		network_name_input = input()
		save_network(network_name_input)

if new_or_load_input == '2': # Load network from file
	print('Enter name of file for load')
	network_name_input = input()
	f = open(str(network_name_input) + '.txt') # create new file if none
	neurons, W = [], []
	for line in f:
		if line[0] == '#': # it is description
			print(line)
		if line[0] == 'N': # it is neuron
			neuron = line.split('/')
			neuron_fun = neuron[1]
			neuron_id = int(neuron[2])
			neuron_act_f_r = float(neuron[3])
			if neuron_fun == 'R': # attention! ThIS IS NOT GOOD FIX
				globals()[neuron_fun+'1'] = Neuron() # create new neuron object
				globals()[neuron_fun].append(globals()[neuron_fun+'1']) # append to list
			else: # in fact every neuron has to have not id number in var name
				globals()[neuron_fun+'%s' % neuron_id] = Neuron() # create new neuron object
				globals()[neuron_fun].append(globals()[neuron_fun+'%s' % neuron_id]) # append to list
			globals()[neuron_fun][-1].fun = neuron_fun # set function of last element of neurons list
			globals()[neuron_fun][-1].id = neuron_id
		if line[0] == 'S': # it is sinapsis
			sinapsis = line.split('/')
			sinapsis_id = int(sinapsis[1])
			connection_from = int(sinapsis[2])
			connection_to = int(sinapsis[3])
			weight = float(sinapsis[4])
			globals()['W%s' % sinapsis_id] = Sinapse() # create new sinapse object
			W[-1].id = sinapsis_id
			for n in neurons:
				if n.id == connection_from:
					W[-1].connection_from = n
				if n.id == connection_to:
					W[-1].connection_to = n
			W[-1].weight = weight
	f.close()
	print('Network has been load')
	# print_network()
	control_question()
	new_questions_control()
	# start_time = time.time()
	# result = learn() # errors rate after learning
	# n_q_result = new_questions_control(True) # errors on new questions
	# print_process(i, result, n_q_result, start_time)

if new_or_load_input == '3': # Best of five
	for bof in range(times_of_network_intialize):
		total_errors = 0
		total_errors_for_new_questions = 0
		for w in W:
			w.random_weight() # reset sinapses weight
		learn_every_epoch()
		control_question()
		new_questions_control()
	os.system('say "Winner has been found..."')

