import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

# ==============================================================================
# SECCIÓN 1: ENTRADA DE DATOS DEL USUARIO 
# ==============================================================================

def obtener_datos_lp():
    #Inicio de procedimeinto para ingrear problema de programación lineal orientado a la maximización.
    
    print("\n Introduzca los coeficientes correspondientes de la función objetivo [Z], separados por espacios.")
    print("Ejemplo: para Z = 3x1 + 5x2, escribe: 3 5")
    coeficientes_str = input("Coeficientes: ").split()
    funcion_objetivo = np.array([float(c) for c in coeficientes_str])
    num_variables = len(funcion_objetivo)

    
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
    
    return variable_names, funcion_objetivo, restricciones

# ==============================================================================
# SECCIÓN 2: LÓGICA DEL MÉTODO SIMPLEX
# ==============================================================================

def resolver_simplex(variable_names, funcion_objetivo, restricciones):
    """
    Resuelve un problema de programación lineal de MAXIMIZACIÓN usando el método Simplex.
    """
    
    num_variables = len(funcion_objetivo)
    num_restricciones = len(restricciones)
    
    num_cols_tabla = num_variables + num_restricciones + 1
    tabla = np.zeros((num_restricciones + 1, num_cols_tabla))

    for i in range(num_restricciones):
        tabla[i, :num_variables] = restricciones[i]['coeficientes']
        tabla[i, num_variables + i] = 1
        tabla[i, -1] = restricciones[i]['rhs']

    tabla[-1, :num_variables] = -funcion_objetivo
    
    
    column_labels = variable_names + [f's{i+1}' for i in range(num_restricciones)] + ['RHS']

    print("\n--- tabla Inicial ---")
    print(pd.DataFrame(tabla, columns=column_labels))
    print("-" * 50)
    
    iteracion = 1
    while np.any(tabla[-1, :-1] < 0):
        print(f"\n--- Iteración #{iteracion} ---")
        
        columna_piv = np.argmin(tabla[-1, :-1])
        print(f"La variable que entra a la base es: {column_labels[columna_piv]}")

        rhs = tabla[:-1, -1]
        col_piv = tabla[:-1, columna_piv]
        
        fila_piv = -1
        min_cociente = float('inf')

        for i in range(num_restricciones):
            if col_piv[i] > 1e-6:
                cociente = rhs[i] / col_piv[i]
                if cociente < min_cociente:
                    min_cociente = cociente
                    fila_piv = i
        
        if fila_piv == -1:
            print("Error: El problema es no acotado.")
            return

        print(f"La fila piv es la fila #{fila_piv + 1}")
        
        elemento_piv = tabla[fila_piv, columna_piv]
        tabla[fila_piv, :] /= elemento_piv
        
        for i in range(num_restricciones + 1):
            if i != fila_piv:
                factor = tabla[i, columna_piv]
                tabla[i, :] -= factor * tabla[fila_piv, :]
        
        print("\n--- Tabla Actualizada ---")
        print(pd.DataFrame(tabla, columns=column_labels))
        print("-" * 50)
        iteracion += 1

    print("\n--- Fin del Algoritmo Simplex (Solución Óptima) ---")
    print("\n--- tabla Final ---")
    print(pd.DataFrame(tabla, columns=column_labels))
    
    print("\n--- Resultados ---")
    valor_optimo_z = tabla[-1, -1]
    print(f"Valor óptimo de Z = {valor_optimo_z:.4f}")

    for i in range(num_variables):
        columna = tabla[:, i]
        es_basica = (np.count_nonzero(columna) == 1) and (np.sum(columna) == 1)
        
        if es_basica:
            fila_del_uno = np.where(columna == 1)[0][0]
            valor_variable = tabla[fila_del_uno, -1]
            print(f"{column_labels[i]} = {valor_variable:.4f}")
        else:
            print(f"{column_labels[i]} = 0.0000")

# ==============================================================================
# SECCIÓN 3: LÓGICA DEL MÉTODO GRÁFICO
# ==============================================================================

def resolver_grafico(variable_names, funcion_objetivo, restricciones):
    """
    Resolución por medio del método gráfico, exclusivo para 2 variables.
    """
    
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
    print(f"Se alcanza en el punto {variable_names[0]} = {punto_optimo[0]:.2f}, {variable_names[1]} = {punto_optimo[1]:.2f}")

    vertices_np = np.array(vertices_factibles)
    ax.scatter(vertices_np[:, 0], vertices_np[:, 1], c='red', zorder=5)
    ax.scatter(punto_optimo[0], punto_optimo[1], c='blue', s=100, zorder=6, label='Punto Óptimo')
    
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
# SECCIÓN 4: MENÚ PRINCIPAL Y EJECUCIÓN
# ==============================================================================

def menu():
    """
    Muestra el menú principal y dirige el flujo de la aplicación.
    """
    variable_names, funcion_objetivo, restricciones = obtener_datos_lp()
    
    print("\n--- Resumen del Problema Ingresado ---")
    print("Tipo de Problema: Maximización")

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
            
            resolver_simplex(variable_names, funcion_objetivo, restricciones)
            break
        elif opcion == '2':
            if len(funcion_objetivo) == 2:
                print("\nIniciando solución con Método Gráfico...")
                
                resolver_grafico(variable_names, funcion_objetivo, restricciones)
                break
            else:
                print("Error: El método gráfico solo se puede usar con 2 variables.")
        else:
            print("Opción invalida. Intente de nuevo.")

if __name__ == "__main__":
    menu()