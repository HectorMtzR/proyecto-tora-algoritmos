# archivo: metodo_simplex.py

import numpy as np
import pandas as pd

def resolver_simplex(tipo_problema, funcion_objetivo, restricciones):
    """
    Resuelve un problema de programación lineal usando el método Simplex.
    NOTA: Esta implementación inicial se enfoca en problemas de MAXIMIZACIÓN
    con restricciones del tipo '<='.
    """
    if tipo_problema != 'max':
        print("Advertencia: Esta versión simplificada está optimizada para problemas de maximización.")
        # Para problemas de minimización, se multiplica la función objetivo por -1
        # y se procede como si fuera de maximización.
        funcion_objetivo = -funcion_objetivo

    num_variables = len(funcion_objetivo)
    num_restricciones = len(restricciones)

    # --- 1. Construcción del Tableau Inicial ---
    
    # El número total de columnas será: variables originales + variables de holgura + columna RHS
    num_cols_tableau = num_variables + num_restricciones + 1
    tableau = np.zeros((num_restricciones + 1, num_cols_tableau))

    # Llenar la matriz con los coeficientes de las restricciones
    for i in range(num_restricciones):
        # Coeficientes de las variables de decisión
        tableau[i, :num_variables] = restricciones[i]['coeficientes']
        # Coeficientes de las variables de holgura (matriz identidad)
        tableau[i, num_variables + i] = 1
        # Lado derecho (RHS)
        tableau[i, -1] = restricciones[i]['rhs']

    # Llenar la última fila (función objetivo)
    tableau[-1, :num_variables] = -funcion_objetivo
    
    # Crear etiquetas para las columnas para una mejor visualización (usando pandas)
    column_labels = [f'x{i+1}' for i in range(num_variables)] + \
                    [f's{i+1}' for i in range(num_restricciones)] + \
                    ['RHS']

    print("--- Tableau Inicial ---")
    print(pd.DataFrame(tableau, columns=column_labels))
    print("-" * 50)
    
    iteracion = 1

    # --- 2. Proceso Iterativo del Simplex ---
    while np.any(tableau[-1, :-1] < 0):
        print(f"\n--- Iteración #{iteracion} ---")
        
        # --- Encontrar la Columna Pivote ---
        # La columna con el valor más negativo en la fila Z
        columna_pivote = np.argmin(tableau[-1, :-1])
        print(f"La variable que entra a la base es: {column_labels[columna_pivote]}")

        # --- Encontrar la Fila Pivote (Prueba del Mínimo Cociente) ---
        rhs = tableau[:-1, -1]
        col_piv = tableau[:-1, columna_pivote]
        
        fila_pivote = -1 # Inicializar en un valor inválido
        min_cociente = float('inf')

        for i in range(num_restricciones):
            if col_piv[i] > 0: # Solo se consideran denominadores positivos
                cociente = rhs[i] / col_piv[i]
                if cociente < min_cociente:
                    min_cociente = cociente
                    fila_pivote = i
        
        if fila_pivote == -1:
            print("Error: El problema es no acotado. No se puede encontrar una solución.")
            return None

        print(f"La fila pivote es la fila #{fila_pivote + 1}")
        
        # --- Actualizar el Tableau (Operaciones de Fila) ---
        
        # 1. Hacer el elemento pivote igual a 1
        elemento_pivote = tableau[fila_pivote, columna_pivote]
        tableau[fila_pivote, :] /= elemento_pivote
        
        # 2. Hacer los otros elementos de la columna pivote iguales a 0
        for i in range(num_restricciones + 1):
            if i != fila_pivote:
                factor = tableau[i, columna_pivote]
                tableau[i, :] -= factor * tableau[fila_pivote, :]
        
        print("\n--- Tableau Actualizado ---")
        print(pd.DataFrame(tableau, columns=column_labels))
        print("-" * 50)
        iteracion += 1

    # --- 3. Presentación de Resultados ---
    print("\n--- Fin del Algoritmo Simplex ---")
    print("La solución óptima ha sido encontrada.")
    
    print("\n--- Tableau Final ---")
    print(pd.DataFrame(tableau, columns=column_labels))
    
    print("\n--- Resultados ---")
    # El valor óptimo de Z está en la esquina inferior derecha
    valor_optimo_z = tableau[-1, -1]
    # Si originalmente era un problema de minimización, el resultado es el negativo
    if tipo_problema == 'min':
        valor_optimo_z *= -1
        
    print(f"Valor óptimo de Z = {valor_optimo_z:.4f}")

    # Encontrar el valor de las variables de decisión
    for i in range(num_variables):
        columna = tableau[:, i]
        # Una variable básica tiene un solo '1' en su columna y el resto son '0'
        es_basica = (np.count_nonzero(columna) == 1) and (np.sum(columna) == 1)
        
        if es_basica:
            fila_del_uno = np.where(columna == 1)[0][0]
            valor_variable = tableau[fila_del_uno, -1]
            print(f"{column_labels[i]} = {valor_variable:.4f}")
        else:
            # Si no es básica, su valor es 0
            print(f"{column_labels[i]} = 0.0000")