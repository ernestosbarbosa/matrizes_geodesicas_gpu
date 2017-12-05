import time
import numpy

from metaheuristic.abc import ABC 

# @jit annotation for Just in Time compilation in CUDA
from numba import jit

# importacao dos txts e definicao das matrizes
matrix_A = numpy.loadtxt( "data/Ar.txt" )
matrix_Qx = numpy.loadtxt( "data/QX.txt" )
matrix_AT = numpy.transpose( matrix_A )

@jit
def second_order_design() : 
    ang_P_min_value = 1488773600
    ang_P_max_value = 2310673700
    dist_P_min_value = 300427.37
    dist_P_max_value = 488640.63
    station_P_min_value = 2263501.2
    station_P_max_value = 480988880
        
    # montagem dos vetores de maximo e minimo do espaco de procura
    search_space_min = []
    search_space_max = []
    for i in range( 143 ) :  # explicacao do slide usa 1 como primeiro indice. Adaptado para 0 aqui
        if ( i <= 42 or ( i >= 69 and i <= 111 ) ) :
            search_space_min.append( dist_P_min_value )
            search_space_max.append( dist_P_max_value )
        elif ( i <= 68 or ( i >= 112 and i <= 137 ) ) :
            search_space_min.append( ang_P_min_value )
            search_space_max.append( ang_P_max_value )
        else :
            search_space_min.append( station_P_min_value )
            search_space_max.append( station_P_max_value )
    
    mh = ABC()
    mh.function = function_second_order_design
    mh.problem_dimension = 143
    mh.min = numpy.transpose( [search_space_min] )
    mh.max = numpy.transpose( [search_space_max] )
    mh.cycles = 50
    mh.number_of_candidate_solutions = 250
    mh.run()
    print( 'mh.best_result' )
    print( mh.best_result )
    print( 'mh.best_solution' )
    print( mh.best_solution )

    
@jit
def function_second_order_design( solution_P ) :  # a funcao que calcula o resultado de dT * d a partir de P
    matrix_P = numpy.diag( numpy.transpose( solution_P )[0] )  # a partir do vector de solucoes P, cria uma matrix quadrada (143x143) com a diagonal preenchida pelos valores de P. Demais posicoes recebem 0.
    matrix_ATP = numpy.dot( matrix_AT, matrix_P )  # dot: multiplica matrizes
    matrix_Qxm = numpy.dot( matrix_ATP, matrix_A )
    matrix_QxmT = numpy.transpose( matrix_Qxm )
    matrix_Qxm_QxmT = numpy.dot( matrix_Qxm, matrix_QxmT )
    inv_matrix_Qxm_QxmT = numpy.linalg.inv( matrix_Qxm_QxmT )  # calcula a inversa da matriz
    matrix_QRxm = numpy.dot( matrix_QxmT, inv_matrix_Qxm_QxmT )
    matrix_D = numpy.subtract( matrix_QRxm, matrix_Qx )
    vector_d = numpy.diag( matrix_D )  # extrai diagonal da matriz e retorna em formato de vetor
    vector_dT = numpy.transpose( vector_d )  # transpoe vetor
    return numpy.dot( vector_dT, vector_d )
    

if __name__ == '__main__':
    start_time = time.time()
    second_order_design()
    print("--- %s seconds ---" % (time.time() - start_time))
    