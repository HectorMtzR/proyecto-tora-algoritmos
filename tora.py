# TORA Replication App - Custom Variable Names
# --------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

# ==============================================================================
# SECCIÓN 1: ENTRADA DE DATOS DEL USUARIO (MODIFICADO)
# ==============================================================================

def obtener_datos_lp():
    """
    Guía al usuario para que ingrese un problema de programación lineal de MAXIMIZACIÓN,
    incluyendo nombres para las variables.
    """
    print("--- Asistente para la Creación de Problemas de Programación Lineal (Maximización) ---")

    print("\nIntroduce los coeficientes de la función objetivo (Z), separados por espacios.")
    print("Ejemplo: para Z = 3x1 + 5x2, escribe: 3 5")
    coeficientes_str = input("Coeficientes: ").split()
    funcion_objetivo = np.array([float(c) for c in coeficientes_str])
    num_variables = len(funcion_objetivo)

    # MODIFICACIÓN: Pedir los nombres de las variables
    print("\n--- Asignación de Nombres a las Variables ---")
    variable_names = []
    for i in range(num_variables):
        nombre = input(f"Nombre para la variable x{i+1} (ej. Sillas): ")
        variable_names.append(nombre)

    print(f"\nAhora, introduce las restricciones. El problema tiene las variables: {', '.join(variable_names)}.")
    restricciones = []
    i = 1
    while True:
        print(f"\n--- Restricción #{i} ---")
        print("Introduce los coeficientes de la restricción, en el mismo orden que las variables.")
        coef_restr_str = input(f"Coeficientes de la restricción {i}: ").split()

        if len(coef_restr_str) != num_variables:
            print(f"Error: Debes introducir {num_variables} coeficientes.")
            continue

        coef_restr = [float(c) for c in coef_restr_str]

        while True:
            tipo_desigualdad = input("Tipo de desigualdad (<=, >=, =): ")
            if tipo_desigualdad in ['<=', '>=', '=']:
                break
            print("Entrada no válida. Usa '<=', '>=', o '='.")

        lado_derecho_str = input("Introduce el valor del lado derecho (RHS): ")
        lado_derecho = float(lado_derecho_str)
        
        restricciones.append({
            'coeficientes': np.array(coef_restr),
            'desigualdad': tipo_desigualdad,
            'rhs': lado_derecho
        })

        otra_mas = input("\n¿Deseas añadir otra restricción? (s/n): ").lower()
        if otra_mas != 's':
            break
        i += 1
        
    print("\n¡Problema ingresado con éxito!")
    # MODIFICACIÓN: Devolvemos también la lista con los nombres de las variables.
    return variable_names, funcion_objetivo, restricciones

# ==============================================================================
# SECCIÓN 2: LÓGICA DEL MÉTODO SIMPLEX (MODIFICADO)
# ==============================================================================

def resolver_simplex(variable_names, funcion_objetivo, restricciones):
    """
    Resuelve un problema de programación lineal de MAXIMIZACIÓN usando el método Simplex.
    """
    # MODIFICACIÓN: Acepta 'variable_names' como parámetro.
    num_variables = len(funcion_objetivo)
    num_restricciones = len(restricciones)
    
    num_cols_tableau = num_variables + num_restricciones + 1
    tableau = np.zeros((num_restricciones + 1, num_cols_tableau))

    for i in range(num_restricciones):
        tableau[i, :num_variables] = restricciones[i]['coeficientes']
        tableau[i, num_variables + i] = 1
        tableau[i, -1] = restricciones[i]['rhs']

    tableau[-1, :num_variables] = -funcion_objetivo
    
    # MODIFICACIÓN: Usa la lista 'variable_names' para las etiquetas de las columnas.
    column_labels = variable_names + [f's{i+1}' for i in range(num_restricciones)] + ['RHS']

    print("\n--- Tableau Inicial ---")
    print(pd.DataFrame(tableau, columns=column_labels))
    print("-" * 50)
    
    iteracion = 1
    while np.any(tableau[-1, :-1] < 0):
        print(f"\n--- Iteración #{iteracion} ---")
        
        columna_pivote = np.argmin(tableau[-1, :-1])
        print(f"La variable que entra a la base es: {column_labels[columna_pivote]}")

        rhs = tableau[:-1, -1]
        col_piv = tableau[:-1, columna_pivote]
        
        fila_pivote = -1
        min_cociente = float('inf')

        for i in range(num_restricciones):
            if col_piv[i] > 1e-6:
                cociente = rhs[i] / col_piv[i]
                if cociente < min_cociente:
                    min_cociente = cociente
                    fila_pivote = i
        
        if fila_pivote == -1:
            print("Error: El problema es no acotado.")
            return

        print(f"La fila pivote es la fila #{fila_pivote + 1}")
        
        elemento_pivote = tableau[fila_pivote, columna_pivote]
        tableau[fila_pivote, :] /= elemento_pivote
        
        for i in range(num_restricciones + 1):
            if i != fila_pivote:
                factor = tableau[i, columna_pivote]
                tableau[i, :] -= factor * tableau[fila_pivote, :]
        
        print("\n--- Tableau Actualizado ---")
        print(pd.DataFrame(tableau, columns=column_labels))
        print("-" * 50)
        iteracion += 1

    print("\n--- Fin del Algoritmo Simplex (Solución Óptima) ---")
    print("\n--- Tableau Final ---")
    print(pd.DataFrame(tableau, columns=column_labels))
    
    print("\n--- Resultados ---")
    valor_optimo_z = tableau[-1, -1]
    print(f"Valor óptimo de Z = {valor_optimo_z:.4f}")

    for i in range(num_variables):
        columna = tableau[:, i]
        es_basica = (np.count_nonzero(columna) == 1) and (np.sum(columna) == 1)
        
        if es_basica:
            fila_del_uno = np.where(columna == 1)[0][0]
            valor_variable = tableau[fila_del_uno, -1]
            print(f"{column_labels[i]} = {valor_variable:.4f}")
        else:
            print(f"{column_labels[i]} = 0.0000")

# ==============================================================================
# SECCIÓN 3: LÓGICA DEL MÉTODO GRÁFICO (MODIFICADO)
# ==============================================================================

def resolver_grafico(variable_names, funcion_objetivo, restricciones):
    """
    Resuelve un problema de programación lineal de dos variables de MAXIMIZACIÓN.
    """
    # MODIFICACIÓN: Acepta 'variable_names' como parámetro.
    print("\n--- Iniciando Método Gráfico ---")
    
    d = np.linspace(-1, 50, 500)
    x1, x2 = np.meshgrid(d, d)
    fig, ax = plt.subplots()
    condiciones_factibles = (x1 >= 0) & (x2 >= 0)

    print("\n--- Restricciones ---")
    for i, r in enumerate(restricciones):
        coefs = r['coeficientes']
        rhs = r['rhs']
        condicion_str = f"{coefs[0]}*x1 + {coefs[1]}*x2 {r['desigualdad']} {rhs}"
        condiciones_factibles &= eval(condicion_str)
        # MODIFICACIÓN: Usa los nombres de variables en la descripción de la restricción.
        print(f"R{i+1}) {coefs[0]} {variable_names[0]} + {coefs[1]} {variable_names[1]} {r['desigualdad']} {rhs}")

        if coefs[1] != 0:
            x2_line = (rhs - coefs[0] * d) / coefs[1]
            ax.plot(d, x2_line, label=f'R{i+1}')
        else:
            ax.axvline(x=rhs / coefs[0], label=f'R{i+1}')

    ax.imshow(condiciones_factibles.astype(int), extent=(x1.min(), x1.max(), x2.min(), x2.max()),
              origin="lower", cmap="Greens", alpha=0.3)

    ejes = [{'coeficientes': np.array([1, 0]), 'rhs': 0}, {'coeficientes': np.array([0, 1]), 'rhs': 0}]
    # (El resto de la lógica de cálculo de vértices no necesita cambios)
    todas_las_lineas = restricciones + ejes
    puntos_interseccion = []
    #... (código sin cambios)
    for line1, line2 in combinations(todas_las_lineas, 2):
        try:
            punto = np.linalg.solve(np.array([line1['coeficientes'], line2['coeficientes']]), np.array([line1['rhs'], line2['rhs']]))
            puntos_interseccion.append(punto)
        except np.linalg.LinAlgError:
            continue
    vertices_factibles = []
    for p in puntos_interseccion:
        if p[0] < -1e-6 or p[1] < -1e-6: continue
        es_factible = True
        for r in restricciones:
            valor_evaluado = np.dot(r['coeficientes'], p)
            if (r['desigualdad'] == '<=' and valor_evaluado > r['rhs'] + 1e-6) or (r['desigualdad'] == '>=' and valor_evaluado < r['rhs'] - 1e-6) or (r['desigualdad'] == '=' and not np.isclose(valor_evaluado, r['rhs'])):
                es_factible = False
                break
        if es_factible and not any(np.allclose(p, v) for v in vertices_factibles):
            vertices_factibles.append(p)

    if not vertices_factibles:
        print("\nNo se encontró una región factible.")
        plt.show()
        return

    print("\n--- Evaluación de la Función Objetivo en cada Vértice Factible ---")
    mejor_valor = -float('inf')
    punto_optimo = None
    for v in vertices_factibles:
        valor_z = np.dot(funcion_objetivo, v)
        print(f"En el vértice ({v[0]:.2f}, {v[1]:.2f}), Z = {valor_z:.2f}")
        if valor_z > mejor_valor:
            mejor_valor = valor_z
            punto_optimo = v

    print("\n--- Solución Óptima ---")
    print(f"El valor óptimo de Z es {mejor_valor:.2f}")
    # MODIFICACIÓN: Usa los nombres de las variables en la solución final.
    print(f"Se alcanza en el punto {variable_names[0]} = {punto_optimo[0]:.2f}, {variable_names[1]} = {punto_optimo[1]:.2f}")

    vertices_np = np.array(vertices_factibles)
    ax.scatter(vertices_np[:, 0], vertices_np[:, 1], c='red', zorder=5)
    ax.scatter(punto_optimo[0], punto_optimo[1], c='blue', s=100, zorder=6, label='Punto Óptimo')
    
    # MODIFICACIÓN: Usa los nombres de las variables en los ejes del gráfico.
    ax.set_xlabel(variable_names[0])
    ax.set_ylabel(variable_names[1])
    ax.legend()
    ax.grid(True)
    max_x = max(v[0] for v in vertices_factibles) + 5
    max_y = max(v[1] for v in vertices_factibles) + 5
    ax.set_xlim(left=-1, right=max_x)
    ax.set_ylim(bottom=-1, top=max_y)

    plt.show()

# ==============================================================================
# SECCIÓN 4: MENÚ PRINCIPAL Y EJECUCIÓN (MODIFICADO)
# ==============================================================================

def menu_principal():
    """
    Muestra el menú principal y dirige el flujo de la aplicación.
    """
    # MODIFICACIÓN: La función de entrada ahora devuelve 3 valores.
    variable_names, funcion_objetivo, restricciones = obtener_datos_lp()
    
    print("\n--- Resumen del Problema Ingresado ---")
    print("Tipo de Problema: Maximización")
    # MODIFICACIÓN: Muestra los nombres de las variables en el resumen.
    z_str = "Z = " + " + ".join([f"{c} {variable_names[i]}" for i, c in enumerate(funcion_objetivo)])
    print(f"Función Objetivo: {z_str}")
    print("Sujeto a las siguientes restricciones:")
    for i, r in enumerate(restricciones):
        r_str = " + ".join([f"{c} {variable_names[j]}" for j, c in enumerate(r['coeficientes'])])
        print(f"  {i+1}) {r_str} {r['desigualdad']} {r['rhs']}")

    print("\n--- ¿Qué método deseas utilizar? ---")
    print("1. Método Simplex")
    print("2. Método Gráfico (solo si el problema tiene 2 variables)")
    
    while True:
        opcion = input("Selecciona una opción (1 o 2): ")
        if opcion == '1':
            print("\nIniciando solución con Método Simplex...")
            # MODIFICACIÓN: Pasa 'variable_names' a la función.
            resolver_simplex(variable_names, funcion_objetivo, restricciones)
            break
        elif opcion == '2':
            if len(funcion_objetivo) == 2:
                print("\nIniciando solución con Método Gráfico...")
                # MODIFICACIÓN: Pasa 'variable_names' a la función.
                resolver_grafico(variable_names, funcion_objetivo, restricciones)
                break
            else:
                print("Error: El método gráfico solo se puede usar con 2 variables.")
        else:
            print("Opción no válida. Inténtalo de nuevo.")

if __name__ == "__main__":
    menu_principal()