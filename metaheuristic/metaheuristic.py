
from numba import vectorize, float64, float32

class Metaheuristic():

    def __init__( self ) :
        self._problem_dimension = 0
        self._cycles = 0
        self._cycles_performed = 0
        self._number_of_candidate_solutions = 0
        self._function = None
        self._best_solution = []
        self._best_result = None
        self._best_solution_history = []
        self._best_result_history = []
        self._number_of_fitness_evaluation = 0

    def define_search_space( self, search_space_dict ) :
        self.min = search_space_dict['min']
        self.max = search_space_dict['max']

    def evaluate_fitness( self, value ) :
        self.number_of_fitness_evaluation += 1
        return self.ret_abs(value)

    # save the variable containing the function to be executed by the MH algorithm
    def set_function( self, function ):
        self.function = function

    @property
    def problem_dimension( self ):
        return self._problem_dimension

    @problem_dimension.setter
    def problem_dimension( self, problem_dimension ):
        self._problem_dimension = problem_dimension

    @property
    def cycles( self ):
        return self._cycles

    @cycles.setter
    def cycles( self, cycles ):
        self._cycles = cycles

    @property
    def cycles_performed( self ):
        return self._cycles_performed

    @cycles_performed.setter
    def cycles_performed( self, cycles_performed ):
        self._cycles_performed = cycles_performed

    @property
    def number_of_candidate_solutions( self ):
        return self._number_of_candidate_solutions

    @number_of_candidate_solutions.setter
    def number_of_candidate_solutions( self, number_of_candidate_solutions ):
        self._number_of_candidate_solutions = number_of_candidate_solutions

    @property
    def best_solution( self ):
        return self._best_solution

    @best_solution.setter
    def best_solution( self, best_solution ):
        self._best_solution = best_solution

    @property
    def best_result( self ):
        return self._best_result

    @best_result.setter
    def best_result( self, best_result ):
        self._best_result = best_result

    @property
    def best_solution_history( self ):
        return self._best_solution_history

    @best_solution_history.setter
    def best_solution_history( self, best_solution_history ):
        self._best_solution_history = best_solution_history

    @property
    def best_result_history( self ):
        return self._best_result_history

    @best_result_history.setter
    def best_result_history( self, best_result_history ):
        self._best_result_history = best_result_history

    @property
    def number_of_fitness_evaluation( self ):
        return self._number_of_fitness_evaluation

    @number_of_fitness_evaluation.setter
    def number_of_fitness_evaluation( self, number_of_fitness_evaluation ):
        self._number_of_fitness_evaluation = number_of_fitness_evaluation

    # CUDA Functions
    @vectorize([
        float32(float32),
        float64(float64)
    ])
    def ret_abs(value):
        return 1 / ( 1 + abs( value ) )
