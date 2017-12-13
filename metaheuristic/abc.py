from copy import deepcopy

import numpy

from metaheuristic.metaheuristic import Metaheuristic

# @jit annotation for Just in Time compilation in CUDA
from numba import jit, vectorize, float32, float64

class ABC( Metaheuristic ):
    
    def run( self ) :
        self._best_result_overall = False
        self._best_solution_overall = False
        # here, number_of_candidate_solutions is the total colony size
        self._solution_abandonment_limit = int( ( self.number_of_candidate_solutions / 2 ) * self.problem_dimension )
        self._food_amount = int( self.number_of_candidate_solutions / 2 )
        # initialize arrays positions
        self._result_of_foods = [ 0 for _ in range( self.food_amount ) ]
        self._fitness_of_foods = [ 0 for _ in range( self.food_amount ) ]
        self._foods = []
        for _ in range( self.food_amount ) :
            self._foods.append( numpy.transpose( [numpy.zeros( self.problem_dimension )] ) )
        self._probabilities = numpy.zeros( self.food_amount )
        self._improv_attempt_counter = [ 0 for i in range( self.food_amount ) ]
        for i in range( self.food_amount ) :
            self.generate_new_food( i )  
        self.evaluate_solutions()
        self.memorize_best_solution()
        for i in range( self.cycles ) :
            if i % 25 == 0 :
                print( '\rABC Cycle ' + str( i ) + '/' + str( self.cycles ), end = '' )
            self.employed_bees_phase()
            self.evaluate_probabilities()
            self.onlooker_bees_phase ()
            self.memorize_best_solution()
            self.scout_bees_phase()
            self.cycles_performed += 1
        if abs( self.best_result_overall ) < abs( self.best_result ) :
            self.best_result = self.best_result_overall
            self.best_solution = self.best_solution_overall
        print( '\rABC Cycle ' + str( self.cycles ) + '/' + str( self.cycles ) )
        print( '==Finished==' )
        
    def scout_bees_phase( self ):
        food_to_replace_index = self.improv_attempt_counter.index( max( self.improv_attempt_counter ) )
        if self.improv_attempt_counter[ food_to_replace_index ] >= self.solution_abandonment_limit :
            self.generate_new_food( food_to_replace_index )
            self.result_of_foods[ food_to_replace_index ] = self.function( self.foods[ food_to_replace_index ] )
            self.fitness_of_foods[ food_to_replace_index ] = self.evaluate_fitness( self.result_of_foods[ food_to_replace_index ] )
            self.memorize_best_solution()

    def check_limits( self, value, dimension ): # check limits based on search space
        if value < self.min[ dimension ] : 
            return self.min[ dimension ]
        elif value > self.max[ dimension ] : 
            return self.max[ dimension ]
        else :
            return value
        
    def memorize_best_solution( self ) :
        max_fitness_index = ( self.fitness_of_foods ).index( max( self.fitness_of_foods ) )
        self.best_solution = self.foods[ max_fitness_index ]
        self.best_result = self.result_of_foods[ max_fitness_index ]
        if ( bool( self.best_result_overall ) == False ) or ( abs( self.best_result_overall ) > abs( self.best_result ) ) :
            self.best_result_overall = self.best_result
            self.best_solution_overall = self.best_solution
            self.best_result_history.append( self.best_result )
            self.best_solution_history.append( self.best_solution )

    @property
    def probabilities( self ):
        return self._probabilities

    @probabilities.setter
    def probabilities( self, probabilities ):
        self._probabilities = probabilities

    @property
    def improv_attempt_counter( self ):
        return self._improv_attempt_counter

    @improv_attempt_counter.setter
    def improv_attempt_counter( self, improv_attempt_counter ):
        self._improv_attempt_counter = improv_attempt_counter

    @property
    def food_amount( self ):
        return self._food_amount

    @food_amount.setter
    def food_amount( self, food_amount ):
        self._food_amount = food_amount
 
    @property
    def solution_abandonment_limit( self ):
        return self._solution_abandonment_limit

    @solution_abandonment_limit.setter
    def solution_abandonment_limit( self, solution_abandonment_limit ):
        self._solution_abandonment_limit = solution_abandonment_limit
        
    @property
    def fitness( self ):
        return self._fitness

    @fitness.setter
    def fitness( self, fitness ):
        self._fitness = fitness
        
    @property
    def foods( self ):
        return self._foods

    @foods.setter
    def foods( self, foods ):
        self._foods = foods

    @property
    def result_of_foods( self ):
        return self._result_of_foods

    @result_of_foods.setter
    def result_of_foods( self, result_of_foods ):
        self._result_of_foods = result_of_foods

    @property
    def fitness_of_foods( self ):
        return self._fitness_of_foods

    @fitness_of_foods.setter
    def fitness_of_foods( self, fitness_of_foods ):
        self._fitness_of_foods = fitness_of_foods

    @property
    def best_result_overall( self ):
        return self._best_result_overall

    @best_result_overall.setter
    def best_result_overall( self, best_result_overall ):
        self._best_result_overall = best_result_overall

    @property
    def best_solution_overall( self ):
        return self._best_solution_overall

    @best_solution_overall.setter
    def best_solution_overall( self, best_solution_overall ):
        self._best_solution_overall = best_solution_overall

   #CUDA JIT

    @jit    
    def onlooker_bees_phase( self ):
        t = 0
        i = 0
        while( t < self.food_amount ) :
            if numpy.random.uniform( 0, 1 ) < self.probabilities[i] :
                t += 1
                phi = numpy.random.uniform( -1, 1 )
                j = numpy.random.randint( 0, self.problem_dimension )
                k = numpy.random.randint( 0, self.food_amount )
                while( i == k ) :
                    k = numpy.random.randint( 0, self.food_amount )
                v = deepcopy( self.foods[i] )
                v[j] = self.get_food(self.foods[i][j], self.foods[k][j], phi)
#                v[j] = self.foods[i][j] + phi * ( self.foods[i][j] - self.foods[k][j] )  # change one parameter (j) for the food (v)
                v[j] = self.check_limits( v[j], j )
                v_solution = self.function( v ) 
                v_fitness = self.evaluate_fitness( v_solution )
                if( v_fitness > self.fitness_of_foods[i] ) : # we improved the solution
                    self.improv_attempt_counter[i] = 0
                    self.foods[i] = v
                    self.result_of_foods[i] = v_solution
                    self.fitness_of_foods[i] = v_fitness
                else :
                    self.improv_attempt_counter[i] += 1
            i = i + 1 if i < ( self.food_amount - 1 ) else 0
            
    @jit
    def evaluate_probabilities( self ) :
        fitness_sum = sum( self.fitness_of_foods )
        for i in range( self.food_amount ) :
            self.probabilities[i] = self.fitness_of_foods[i] / fitness_sum

    @jit    
    def employed_bees_phase( self ):
        for i in range( self.food_amount ) :
            phi = numpy.random.uniform( -1, 1 )
            j = numpy.random.randint( 0, self.problem_dimension )
            k = numpy.random.randint( 0, self.food_amount )
            while( i == k ) :
                k = numpy.random.randint( 0, self.food_amount )
            v = deepcopy( self.foods[i] )
#            v[j] = self.foods[i][j] + phi * ( self.foods[i][j] - self.foods[k][j] )  # change one parameter (j) for the food (v)
            v[j] = self.get_food(self.foods[i][j], self.foods[k][j], phi )  # change one parameter (j) for the food (v)
            v[j] = self.check_limits( v[j], j )
            v_result = self.function( v ) 
            v_fitness = self.evaluate_fitness( v_result )
            if v_fitness > self.fitness_of_foods[i] : # we improved the solution
                self.improv_attempt_counter[i] = 0
                self.foods[i] = v
                self.result_of_foods[i] = v_result
                self.fitness_of_foods[i] = v_fitness
            else :
                self.improv_attempt_counter[i] += 1

    @jit        
    def generate_new_food( self, i ) :
        for j in range( self.problem_dimension ) :
            self.foods[i][j] = numpy.random.uniform( self.min[ j ], self.max[ j ] )
        self.improv_attempt_counter[i] = 0

    @jit
    def evaluate_solutions( self ) :
        for i, f in enumerate( self.foods ) :
            self.result_of_foods[ i ] = self.function( f )
            self.fitness_of_foods[ i ] = self.evaluate_fitness( self.result_of_foods[ i ] ) 

    @vectorize([
        float32(float32, float32, float32),
        float64(float64, float64, float64)
    ])
    def get_food(a, b, phi):
        return a + phi * ( a - b )  # change one parameter (j) for the food (v)