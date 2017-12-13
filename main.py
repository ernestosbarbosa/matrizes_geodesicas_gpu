import time
from metaheuristic.abc import ABC 
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
from numba import jit
import numpy as np
np.seterr(invalid='ignore')

# importacao dos txts e definicao das matrizes
matrix_A = np.loadtxt( "data/Ar.txt" )
matrix_Qx = np.loadtxt( "data/QX.txt" )
matrix_AT = np.transpose( matrix_A )
BLOCK_SIZE = 16

# compila o codigo cuda
mod = SourceModule(open("matrix.cu", "r").read())
matrixtrans = mod.get_function("gpu_matrix_transpose")
matrixmul = mod.get_function("gpu_matrix_mult")

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
    mh.min = np.transpose( [search_space_min] )
    mh.max = np.transpose( [search_space_max] )
    mh.cycles = 50
    mh.number_of_candidate_solutions = 250
    mh.run()
    print( 'mh.best_result' )
    print( mh.best_result )
    print( 'mh.best_solution' )
    print( mh.best_solution )
   
def function_second_order_design( solution_P ) :  # a funcao que calcula o resultado de dT * d a partir de P
    matrix_P = np.diag( np.transpose( solution_P )[0] )  # a partir do vector de solucoes P, cria uma matrix quadrada (143x143) com a diagonal preenchida pelos valores de P. Demais posicoes recebem 0.
    matrix_ATP = np.dot( matrix_AT, matrix_P )  
    #matrix_ATP2 = matrix_mult( matrix_AT, matrix_P )  
    #print(matrix_ATP)
    #print(matrix_ATP2)
    
    #start_time = time.time()
    #matrix_Qxm = dot_py( matrix_ATP, matrix_A )
    matrix_Qxm = np.dot( matrix_ATP, matrix_A )
    matrix_Qxm2 = matrix_mult( matrix_ATP, matrix_A )  
    np.savetxt('resultDot.txt', matrix_Qxm, fmt='%.2f')
    #print(matrix_Qxm[0][0])
    #print(matrix_Qxm2[0][0])

    #matrix_QxmT = matrixTranspose( matrix_Qxm )
    matrix_QxmT = np.transpose( matrix_Qxm )
    #matrix_QxmT = matrix_transpose( matrix_Qxm )

    #print(matrix_QxmT)

    matrix_Qxm_QxmT = dot_py( matrix_Qxm, matrix_QxmT )
    #matrix_Qxm_QxmT = np.dot( matrix_Qxm, matrix_QxmT )
    #matrix_Qxm_QxmT = matrix_mult( matrix_Qxm, matrix_QxmT )  

    inv_matrix_Qxm_QxmT = np.linalg.inv( matrix_Qxm_QxmT )  # calcula a inversa da matriz

    matrix_QRxm = dot_py( matrix_QxmT, inv_matrix_Qxm_QxmT )
    #matrix_QRxm = np.dot( matrix_QxmT, inv_matrix_Qxm_QxmT )
    #matrix_QRxm = matrix_mult( matrix_QxmT )  
    #print("--- %s seconds ---" % (time.time() - start_time))
    
    matrix_D = np.subtract( matrix_QRxm, matrix_Qx )
    
    vector_d = np.diag( matrix_D )  # extrai diagonal da matriz e retorna em formato de vetor
    vector_dT = np.transpose( vector_d )  # transpoe vetor
    return np.dot( vector_dT, vector_d )



#matriz transposta
def matrix_transpose(mat_in):

    a_gpu = gpuarray.to_gpu(np.array(mat_in)) 
    mat_out = gpuarray.zeros((len(mat_in), len(mat_in[0])), np.float32)

    # chama a funcao
    matrixtrans(a_gpu, mat_out, block = (BLOCK_SIZE, BLOCK_SIZE, 1) )

    return mat_out.get()

#multiplicação matrizes
def matrix_mult(a, b):
    np.savetxt('resultAC.txt', a, fmt='%.2f')
    np.savetxt('resultBC.txt', b, fmt='%.2f')

    m = np.int32(len(a));
    n = np.int32(len(a[0]));
    k = np.int32(len(b[0]));

    # passa matriz da memoria da CPU para a GPU 
    a_gpu = gpuarray.to_gpu(a) 
    np.savetxt('resultAG.txt', a_gpu.get(), fmt='%.2f')
    b_gpu = gpuarray.to_gpu(b)
    np.savetxt('resultBG.txt', b_gpu.get(), fmt='%.2f')

    # criando matriz resultado (C = A * B) | tam = m*k
    c_gpu = gpuarray.zeros((m , k), np.float32)

    # chama a funcao
    matrixmul( a_gpu, b_gpu, c_gpu, m, n, k, 
    block = (BLOCK_SIZE, BLOCK_SIZE, 1) )
    
    np.savetxt('resulGPU.txt', c_gpu.get(), fmt='%.2f')
    return c_gpu.get()


#multiplicação matrizes numba
@jit
def dot_py(A,B):
    m, n = A.shape
    p = B.shape[1]
    C = np.zeros((m,p))
    for i in range(0,m):
        for j in range(0,p):
            for k in range(0,n):
                C[i,j] += A[i,k]*B[k,j] 
    return C

#matriz transposta numba
@jit
def matrixTranspose(anArray):
    transposed = np.zeros((len(anArray[0]),len(anArray)))
    for t in range(len(anArray)):
        transposed[t] = [0]*len(anArray)
        for tt in range(len(anArray[t])):
            transposed[t][tt] = anArray[tt][t]
    return transposed


if __name__ == '__main__':
    start_time = time.time()
    second_order_design()
    print("--- %s seconds ---" % (time.time() - start_time))
    