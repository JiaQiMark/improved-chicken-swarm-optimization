import numpy as np
import math
import matplotlib.pyplot as plt



class Chicken:
    # Defining Constructor
    def __init__(self):
        # Sets the Basic Binary String of Length to The Number of Features,
        # This is needed to Compare the New Generation with the Existting One ,
        # And Changing The Orderof Fitness
        self.current_position = np.random.uniform(-search_range_num, search_range_num, Dn)
        self.current_fitness = fitness_function(self.current_position)  # Inititally Not Evaluating the Fitness
        self.optimal_positioin = self.current_position
        self.optimal_fitness = self.current_fitness

        self.group = -1  # Inirially Not Evaluating Any Group
        self.species_name = "none"  # Later Will change to Rooster, Chicken or Hen
        self.a = 6
        self.b = 0.3

    def evaluate(self):
        if self.current_fitness < self.optimal_fitness:
            self.optimal_positioin = self.current_position
            self.optimal_fitness   = self.current_fitness

    '''
        Group of Functions which need to update the position of Chickens.
        Note : The Position will be first stored in a different property ie next_position_to_which_chicken_will_move will store the next address . Moving on the lane , The Fitness Count Obtained from new Generation will help in Updating the solution
    '''

    '''
        All the Functions will take in a parameter as The Number of Groups the Population is Divied into , For Example , If the Total Population is 10 , The best suited Group will be 10/5 , ie 2. All the Roosters will be then updated to the count of the following Appropriate Distribution
    '''

    # Integer , Class Rooster
    def update_location_rooster(self, number_of_groups, rooster, iterations):
        inertia_weight = w_max - (w_max - w_min) * (1 / (1 + np.exp(self.a - self.b * iterations)))

        # Example as Like if the Population is divided into 2 Groups ,
        # and then Total Option is Limited to either 0 or 1 , 0 for the First Group and 1 For Other Group
        random_number_groups = np.random.randint(0, number_of_groups)

        while rooster[random_number_groups].group == self.group:
            ## Checking If It doesn't Belong to the same Group
            random_number_groups = np.random.randint(0, number_of_groups)

        ### Evaluating The Equation According to The Algorithm
        ## Initalizing Sigma
        sigma_square = 0
        e = 1e-20
        if rooster[random_number_groups].group != self.group:
            if rooster[random_number_groups].current_fitness >= self.current_fitness:
                sigma_square = 1
            else:
                sigma_square = np.exp((rooster[random_number_groups].current_fitness - self.current_fitness) /
                                      (np.abs(self.current_fitness) + e))

        # Create Gaussian Distribution  with Mean 0 and Standard Deviation is sigma_sqare
        random_distribution = np.random.normal(0, sigma_square)

        '''
            We are Only Updating The Next Position , And Not the Original Position , 
            Because the Update is Valid only when The Original Fitness is found lowered to The Mutated Fitness
        '''

        for index in range(0, Dn):
            self.current_position[index] =\
                  inertia_weight * self.current_position[index] +\
                  (random_distribution * (self.optimal_positioin[index] - self.current_position[index]))
            self.current_position[index] = np.clip(self.current_position[index], -search_range_num, search_range_num)

    def update_location_hen(self, number_of_groups, rooster, iterations):  # Integer , Class Rooster
        inertia_weight = w_max - (w_max - w_min) * (1 / (1 + np.exp(self.a - self.b * iterations)))

        fitness_rooster_1 = None
        position_rooster_1 = None

        for index in range(0, number_of_groups):
            if rooster[index].group == self.group:
                position_rooster_1 = rooster[index].current_position  # Same Group Rooster Position
                fitness_rooster_1 = rooster[index].current_fitness    # Same Group Rooster Health

        random_number_of_groups = np.random.randint(0, number_of_groups)

        while rooster[random_number_of_groups].group == self.group:
            ## More not Getting the same Rooster Group
            random_number_of_groups = np.random.randint(0, number_of_groups)


        fitness_current_hen = self.current_fitness  # Fitness of Current Hen
        position_current_hen = self.current_position  # Position of Current Hen
        e = 1e-20  # Defining the Smallest Constant

        # Defining S1 and S2 For The Parameters Listed
        s1 = np.exp((fitness_current_hen - fitness_rooster_1) / (np.abs(fitness_current_hen) + e))
        s3 = np.exp((self.optimal_fitness - fitness_current_hen))

        # Defining a Uniform Random Number Between 0 and 1
        uniform_random_number_between_0_and_1 = np.random.rand()

        # Note , Changing the next position and not the original position for Comparing different fitness
        for index in range(0, Dn):
            self.current_position[index] = (
                    inertia_weight * position_current_hen[index]
                    + s1 * uniform_random_number_between_0_and_1 * (
                                position_rooster_1[index] - position_current_hen[index])
                    + s3 * uniform_random_number_between_0_and_1 * (
                                self.optimal_positioin[index] - position_current_hen[index]))
            self.current_position[index] = np.clip(self.current_position[index], -search_range_num, search_range_num)



def implementing_cso(population, individuals_in_group, maximum_generation, FL=0.5):
    # Initializing the total number of Groups for the Population ,
    # Appropriate Will be Population in Multiple of 10's and Dividing It in Multiple of 5
    number_of_groups = int(population / individuals_in_group)
    print("The Number Of Group The Swarm Is Divided : ", number_of_groups)

    population_list = []  # List Storing the Object of Chicken.
    optimal_solution_fitness_list = []
    for index in range(population):
        population_list.append(Chicken())
        population_list[index].evaluate()

    iteration_test_cases = 0
    while iteration_test_cases < maximum_generation:
        # 更新鸡群的hierarchy
        population_list.sort(key=lambda c: c.current_fitness, reverse=False)
        rooster_class = population_list[:number_of_groups]

        for index in range(number_of_groups):
            population_list[index].species_name = "Rooster"  # Example of 10, First  2 being Roosters
            population_list[index].group = index
        for index in range(number_of_groups, population):
            population_list[index].species_name = "Hen"      # Example of 10 , index of 2,3,4,5 being Hens
            population_list[index].group = index % individuals_in_group

        # It Starts Here!!!!
        for c_id in range(0, population):
            if population_list[c_id].species_name == "Rooster":
                population_list[c_id].update_location_rooster(number_of_groups, rooster_class, iteration_test_cases)
            elif population_list[c_id].species_name == "Hen":
                population_list[c_id].update_location_hen(number_of_groups, rooster_class, iteration_test_cases)
            population_list[c_id].evaluate()

        population_list.sort(key=lambda c: c.current_fitness, reverse=False)
        iteration_test_cases += 1

        # 在每次迭代的最后一步，记录当前迭代的最优值
        fitness_value = fitness_function(population_list[0].optimal_positioin)
        optimal_solution_fitness_list.append(fitness_value)
        print(iteration_test_cases, "/", maximum_generation, "    optimal:", fitness_value)

    return optimal_solution_fitness_list


w_min = 0.1
w_max = 0.7

from fitness_function import fitness_function
from fitness_function import Dn, population_size, group_size, max_iterations, search_range_num

optimal_solution = implementing_cso(population_size, group_size, max_iterations, 0.5)
iterations = np.linspace(0, max_iterations-1, len(optimal_solution), dtype=int)

plt.xlabel('iterations')
plt.ylabel('fitness')
plt.title('ascso')
plt.plot(iterations, optimal_solution)
plt.show()





