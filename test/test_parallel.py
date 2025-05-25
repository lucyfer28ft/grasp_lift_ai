from joblib import Parallel, delayed
import time

def tarea_larga(i):
    time.sleep(2)
    return i

if __name__ == '__main__':
    resultados = Parallel(n_jobs=-1)(
        delayed(tarea_larga)(i) for i in range(12)
    )
    print(resultados)
