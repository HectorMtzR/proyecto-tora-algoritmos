# archivo: metodo_grafico.py

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def resolver_grafico(tipo_problema, funcion_objetivo, restricciones):
    """
    Resuelve un problema de programación lineal de dos variables
    usando el método gráfico.
    """
    print("\n--- Iniciando Método Gráfico ---")
    
    # --- 1. Preparar el espacio para el gráfico ---
    # Generamos un rango de valores para x1 (eje x)
    d = np.linspace(-1, 50, 500) 
    x1, x2 = np.meshgrid(d, d)

    # Creamos la figura y los ejes para el plot
    fig, ax = plt.subplots()
    
    # Añadimos la restricción de no negatividad (x1 >= 0, x2 >= 0)
    # y la aplicamos a toda la malla del gráfico.
    condiciones_factibles = (x1 >= 0) & (x2 >= 0)

    # --- 2. Dibujar las líneas de las restricciones y definir la región factible ---
    print("\n--- Restricciones ---")
    for i, r in enumerate(restricciones):
        coefs = r['coeficientes']
        rhs = r['rhs']
        
        # Construimos la condición booleana para la región factible
        # eval() nos permite ejecutar una cadena de texto como código Python
        condicion_str = f"{coefs[0]}*x1 + {coefs[1]}*x2 {r['desigualdad']} {rhs}"
        condiciones_factibles &= eval(condicion_str)
        print(f"R{i+1}) {coefs[0]}x1 + {coefs[1]}x2 {r['desigualdad']} {rhs}")

        # Para dibujar la línea, la tratamos como una igualdad
        # ax1 + bx2 = c  =>  x2 = (c - ax1) / b
        # Se manejan casos donde un coeficiente sea cero (líneas verticales u horizontales)
        if coefs[1] != 0:
            x2_line = (rhs - coefs[0] * d) / coefs[1]
            ax.plot(d, x2_line, label=f'R{i+1}')
        else: # Línea vertical (coeficiente de x2 es 0)
            ax.axvline(x=rhs / coefs[0], label=f'R{i+1}')

    # --- 3. Sombrear la Región Factible ---
    ax.imshow(condiciones_factibles.astype(int), extent=(x1.min(), x1.max(), x2.min(), x2.max()),
              origin="lower", cmap="Greens", alpha=0.3)

    # --- 4. Encontrar los Vértices (Intersecciones) ---
    # Añadimos los ejes (x1=0, x2=0) como si fueran restricciones para encontrar intersecciones
    ejes = [
        {'coeficientes': np.array([1, 0]), 'rhs': 0},  # x1 = 0
        {'coeficientes': np.array([0, 1]), 'rhs': 0}   # x2 = 0
    ]
    todas_las_lineas = restricciones + ejes
    
    # Calculamos la intersección de cada par de líneas
    puntos_interseccion = []
    for line1, line2 in combinations(todas_las_lineas, 2):
        A = np.array([line1['coeficientes'], line2['coeficientes']])
        b = np.array([line1['rhs'], line2['rhs']])
        try:
            # Resolvemos el sistema de ecuaciones 2x2: A*x = b
            punto = np.linalg.solve(A, b)
            puntos_interseccion.append(punto)
        except np.linalg.LinAlgError:
            # Las líneas son paralelas, no hay intersección única
            continue
            
    # --- 5. Filtrar Vértices Factibles ---
    vertices_factibles = []
    print("\n--- Vértices (Puntos de Esquina) ---")
    for p in puntos_interseccion:
        x1_p, x2_p = p
        
        es_factible = True
        # Condición de no-negatividad
        if x1_p < -1e-6 or x2_p < -1e-6: # Usamos una pequeña tolerancia
            es_factible = False
            continue
            
        # Verificar si el punto cumple TODAS las restricciones originales
        for r in restricciones:
            coefs = r['coeficientes']
            rhs = r['rhs']
            desigualdad = r['desigualdad']
            
            valor_evaluado = np.dot(coefs, p)
            
            if desigualdad == '<=' and valor_evaluado > rhs + 1e-6:
                es_factible = False
                break
            elif desigualdad == '>=' and valor_evaluado < rhs - 1e-6:
                es_factible = False
                break
            elif desigualdad == '=' and not np.isclose(valor_evaluado, rhs):
                es_factible = False
                break
        
        if es_factible:
            # Evitar duplicados
            if not any(np.allclose(p, v) for v in vertices_factibles):
                vertices_factibles.append(p)

    # --- 6. Evaluar y Encontrar el Óptimo ---
    if not vertices_factibles:
        print("\nNo se encontró una región factible o es no acotada en la dirección de optimización.")
        return

    print("\n--- Evaluación de la Función Objetivo en cada Vértice Factible ---")
    mejor_valor = None
    punto_optimo = None

    if tipo_problema == 'max':
        mejor_valor = -float('inf')
    else: # min
        mejor_valor = float('inf')

    for v in vertices_factibles:
        valor_z = np.dot(funcion_objetivo, v)
        print(f"En el vértice ({v[0]:.2f}, {v[1]:.2f}), Z = {valor_z:.2f}")
        
        if tipo_problema == 'max' and valor_z > mejor_valor:
            mejor_valor = valor_z
            punto_optimo = v
        elif tipo_problema == 'min' and valor_z < mejor_valor:
            mejor_valor = valor_z
            punto_optimo = v

    # --- 7. Mostrar Resultados y Gráfico ---
    print("\n--- Solución Óptima ---")
    print(f"El valor óptimo de Z es {mejor_valor:.2f}")
    print(f"Se alcanza en el punto x1 = {punto_optimo[0]:.2f}, x2 = {punto_optimo[1]:.2f}")

    # Dibujar los vértices factibles y resaltar el óptimo
    vertices_np = np.array(vertices_factibles)
    ax.scatter(vertices_np[:, 0], vertices_np[:, 1], c='red', zorder=5) # zorder para que se vea encima
    ax.scatter(punto_optimo[0], punto_optimo[1], c='blue', s=100, zorder=6, label='Punto Óptimo')

    # Configuración final del gráfico
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    ax.grid(True)
    
    # Ajustar límites del gráfico para centrarse en la región de interés
    if vertices_factibles:
        max_x = max(v[0] for v in vertices_factibles) + 5
        max_y = max(v[1] for v in vertices_factibles) + 5
        ax.set_xlim(left=-1, right=max_x)
        ax.set_ylim(bottom=-1, top=max_y)

    plt.show()