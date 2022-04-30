import numpy as np

KNIGHTS_MOVESET = [(1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2)]
VALIDATION = False


class KnightTour:
    """
    Consists of a chess_board and each legal knight's move is a neuron.
    each neuron has a state, an output, 2 vertices(which are
    positions on chess_board) and at most 8 neighbours(all the neurons
    that share a vertex with this neuron).
    if a neuron output is 1 then it is in the solution.
    """

    # Initalisation of a KnightTour class (the constructor)
    def __init__(self, board_size):
        self.chess_board_size = board_size
        self.chess_board = []
        for i in range(self.chess_board_size[0]):
            temp = []
            for j in range(self.chess_board_size[1]):
                temp.append(set())
            self.chess_board.append(temp)
        self.neuron_vertices = []
        self.resultant_output = np.array([])
        self.neuron_states = np.array([])
        self.neuron_nearest_neighbours = []
        if VALIDATION:
            print('------first-------')
            self.print_chess_board(self.chess_board)

        self.init()

    # a function which intialises the size of the the chess board
    def print_chess_board(self, board):
        if len(board) == self.chess_board_size[0]:
            for i in range(self.chess_board_size[0]):
                print(board[i])
        else:
            m = 0
            strin = ''
            for i in range(0, len(board), 6):
                print(board[i: i+6])

    # Finds all the possible neurons(knight moves) on the chess_board  and sets the neuron_vertices and neuron
    # neighbours.
    def init(self):
        neuron_temp_counter: int = 0
        # looping through the chess_board
        for first_x in range(self.chess_board_size[0]):
            for first_y in range(self.chess_board_size[1]):
                first_coordinate = first_x * self.chess_board_size[1] + first_y

                for (second_x, second_y) in self.find_neuron_nearest_neighbours((first_x, first_y)):
                    second_coodrinate = second_x * self.chess_board_size[1] + second_y

                    # assigns each neuron with 2 vertices so this is to make
                    # sure that we add the neuron once.
                    if second_coodrinate > first_coordinate:
                        self.chess_board[first_x][first_y].add(neuron_temp_counter)
                        self.chess_board[second_x][second_y].add(neuron_temp_counter)
                        self.neuron_vertices.append({(first_x, first_y), (second_x, second_y)})
                        neuron_temp_counter += 1

        for first_coordinate in range(len(self.neuron_vertices)):
            first_vertex, second_vertex = self.neuron_vertices[first_coordinate]
            # neighbours of neuron i = neighbours of vertex1 + neighbours of vertex2 - i
            neuron_neighbours = self.chess_board[first_vertex[0]][first_vertex[1]].union(self.chess_board[second_vertex[0]][second_vertex[1]]) - {first_coordinate}
            self.neuron_nearest_neighbours.append(neuron_neighbours)

        if VALIDATION:
            print("----init-----")
            print('chess_board')
            self.print_chess_board(self.chess_board)
            print('vertices')
            self.print_chess_board(self.neuron_vertices)
            print('neighbours')
            self.print_chess_board(self.neuron_nearest_neighbours)

    def initialize_neurons(self):
        """
        Initializes each neuron state to 0 and a random number
        between 0 and 1 for neuron outputs.
        """
        self.resultant_output = np.random.randint(2, size=(len(self.neuron_vertices)), dtype=np.int16)
        self.neuron_states = np.zeros((len(self.neuron_vertices)), dtype=np.int16)

        if VALIDATION:
            print('_________initialize_neurons__________________________')
            print('neuron_states:')
            print(self.neuron_states)
            print('outputs')
            print(self.resultant_output)

    def update_neurons_state_output(self):
        """
        Updates the state and output of each neuron.
        """
        neuron_neighbour_summutation = np.zeros((len(self.neuron_states)), dtype=np.int16)
        for i in range(len(self.neuron_nearest_neighbours)):
            neuron_neighbour_summutation[i] = self.resultant_output[list(self.neuron_nearest_neighbours[i])].sum()

        next_state = self.neuron_states + 4 - neuron_neighbour_summutation - self.resultant_output
        # counts the number of changes between the next state and the current state.
        number_of_changes = np.count_nonzero(next_state != self.neuron_states)
        # if next state[i] < 3 ---> output[i] = 0
        # if next state[i] > 0 ---> output[i] = 3
        self.resultant_output[np.argwhere(next_state < 0).ravel()] = 0
        self.resultant_output[np.argwhere(next_state > 3).ravel()] = 1
        self.neuron_states = next_state
        # counts the number of active neurons which are the neurons that their output is 1.
        neuron_activation_counter = len(self.resultant_output[self.resultant_output == 1])

        if VALIDATION:
            print('____________________update________________________')
            print('neuron_states:')
            print(self.neuron_states)
            print('output')
            print(self.resultant_output)

        return neuron_activation_counter, number_of_changes

    def neural_network_closed(self):
        """
        Finds a closed knight's tour.
        """
        even_degree = False
        time = 0
        while True:
            self.initialize_neurons()
            n = 0
            while True:
                neuron_active_counter, change_counter = self.update_neurons_state_output()
                print('_______________info_________________')
                print('active', neuron_active_counter, 'changes', change_counter)
                if change_counter == 0:
                    break
                if self.validate_vertices_degree():
                    even_degree = True
                    break
                n += 1
                if n == 20:
                    break
            time += 1
            if even_degree:
                print('all vertices have degree=2')
                if self.check_hamiltonian_graph():
                    print('solution found!!')
                    self.retrieve_closed_knight_tour_solution()
                    return
                else:
                    even_degree = False

    def check_hamiltonian_graph(self):
        """
        Checks whether the solution is a knight's tour and it's not
        two or more independent hamiltonian graphs.
        """
        # gets the index of active neurons.
        active_neuron_index = np.argwhere(self.resultant_output == 1).ravel()
        # dfs through all active neurons starting from the first element.
        neural_connection = self.validate_closed_knight_tour(neuron=active_neuron_index[0], active_neurons=active_neuron_index)
        if neural_connection:
            return True
        return False

    def validate_closed_knight_tour(self, neuron, active_neurons):
        """
        Performs a DFS algorithm from a starting active neuron
        visiting all active neurons.
        Returns True if the is no active neurons left in the array
        (means we have only on hamiltonian graph).
        """
        # removes the neuron from the active neurons list.
        active_neurons = np.setdiff1d(active_neurons, [neuron])
        # first finds the neighbours of this neuron and then finds which of them are active.
        active_nearest_neighbours = np.intersect1d(active_neurons, list(self.neuron_nearest_neighbours[neuron]))
        # if there was no active neighbours for this neuron, the hamiltonian graph has been
        # fully visited.
        if len(active_nearest_neighbours) is 0:
            # we check if all the active neurons have been visited. if not, it means that there
            # are more than 1 hamiltonian graph and it's not a knight's tour.
            if len(active_neurons) is 0:
                return True
            else:
                return False
        return self.validate_closed_knight_tour(neuron=active_nearest_neighbours[0], active_neurons=active_neurons)

    def retrieve_closed_knight_tour_solution(self):
        """
        Finds and prints the solution.
        """
        visited_neurons = []
        current_vertex = (0, 0)
        labels = np.zeros(self.chess_board_size, dtype=np.int16)
        # gets the index of active neurons.
        active_neuron_index = np.argwhere(self.resultant_output == 1).ravel()
        i = 0
        while len(active_neuron_index) != 0:
            visited_neurons.append(current_vertex)
            labels[current_vertex] = i
            i += 1
            # finds the index of neurons that have this vertex(current_vertex).
            neuron_neighbours = list(self.chess_board[current_vertex[0]][current_vertex[1]])
            # finds the active ones.
            # active neurons that have this vertex are the edges of the solution graph that
            # share this vertex.
            neuron_neighbours = np.intersect1d(neuron_neighbours, active_neuron_index)
            # picks one of the neighbours(the first one) and finds the other vertex of
            # this neuron(or edge) and sets it as the current one
            current_vertex = list(self.neuron_vertices[neuron_neighbours[0]] - {current_vertex})[0]
            # removes the selected neighbour from all active neurons
            active_neuron_index = np.setdiff1d(active_neuron_index, [neuron_neighbours[0]])
        print(labels)

    def active_neurons_vertices(self):
        """
        Returns the vertices of the active neurons(neurons
        that have output=1).
        Used for drawing the edges of the graph in GUI.
        """
        # gets the index of active neurons.
        active_neuron_indices = np.argwhere(self.resultant_output == 1).ravel()
        active_neuron_vertices = []
        for i in active_neuron_indices:
            active_neuron_vertices.append(self.neuron_vertices[i])
        return active_neuron_vertices

    def validate_vertices_degree(self):
        """
        Returns True if all of the vertices have degree=2.
        for all active neurons updates the degree of its
        vertices and then checks if degree has any number
        other than 2.
        """
        # gets the index of active neurons.
        active_neuron_indices = np.argwhere(self.resultant_output == 1).ravel()
        degree = np.zeros((self.chess_board_size[0], self.chess_board_size[1]), dtype=np.int16)

        for i in active_neuron_indices:
            vertex1, vertex2 = self.neuron_vertices[i]
            degree[vertex1[0]][vertex1[1]] += 1
            degree[vertex2[0]][vertex2[1]] += 1

        if VALIDATION:
            print('____________________check degree_______________________')
            print(degree)

        # if all the degrees=2 return True
        if degree[degree != 2].size is 0:
            return True
        return False

    def find_neuron_nearest_neighbours(self, pos):
        """
        Returns all the positions which the knight can move
        giving it's position.
        """
        neighbours_collection = set()
        for (dx, dy) in KNIGHTS_MOVESET:
            new_x, new_y = pos[0]+dx, pos[1]+dy
            if 0 <= new_x < self.chess_board_size[0] and 0 <= new_y < self.chess_board_size[1]:
                neighbours_collection.add((new_x, new_y))
        return neighbours_collection

tour = KnightTour((6, 6))
tour.neural_network_closed()
