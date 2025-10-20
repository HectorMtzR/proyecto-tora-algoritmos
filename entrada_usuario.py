# archivo: entrada_usuario.py

import numpy as np

def obtener_datos_lp():
    """
    Guía al usuario para que ingrese un problema de programación lineal
    y devuelve los datos en un formato estructurado.
    """
    print("--- Asistente para la Creación de Problemas de Programación Lineal ---")

    # 1. Definir tipo de problema (Maximizar o Minimizar)
    while True:
        tipo_problema = input("¿El problema es de Maximización (max) o Minimización (min)?: ").lower()
        if tipo_problema in ['max', 'min']:
            break
        print("Entrada no válida. Por favor, escribe 'max' o 'min'.")

    # 2. Ingresar la Función Objetivo
    print("\nIntroduce los coeficientes de la función objetivo (Z), separados por espacios.")
    print("Ejemplo: para Z = 3x1 + 5x2, escribe: 3 5")
    coeficientes_str = input("Coeficientes: ").split()
    # Convertimos los coeficientes a números de punto flotante
    funcion_objetivo = np.array([float(c) for c in coeficientes_str])
    num_variables = len(funcion_objetivo)

    # 3. Ingresar las Restricciones
    print(f"\nAhora, introduce las restricciones. El problema tiene {num_variables} variables (x1, x2, ...).")
    restricciones = []
    i = 1
    while True:
        print(f"\n--- Restricción #{i} ---")
        print("Introduce los coeficientes de la restricción, separados por espacios.")
        print(f"Ejemplo: para x1 + 2x2 <= 4, escribe: 1 2")
        coef_restr_str = input(f"Coeficientes de la restricción {i}: ").split()

        # Validamos que el número de coeficientes sea correcto
        if len(coef_restr_str) != num_variables:
            print(f"Error: Debes introducir {num_variables} coeficientes, uno para cada variable.")
            continue

        coef_restr = [float(c) for c in coef_restr_str]

        # Pedir el tipo de desigualdad
        while True:
            tipo_desigualdad = input("Tipo de desigualdad (<=, >=, =): ")
            if tipo_desigualdad in ['<=', '>=', '=']:
                break
            print("Entrada no válida. Usa '<=', '>=', o '='.")

        # Pedir el lado derecho (RHS)
        lado_derecho_str = input("Introduce el valor del lado derecho (RHS): ")
        lado_derecho = float(lado_derecho_str)
        
        # Guardamos la restricción como un diccionario para mayor claridad
        restricciones.append({
            'coeficientes': np.array(coef_restr),
            'desigualdad': tipo_desigualdad,
            'rhs': lado_derecho
        })

        # Preguntar si desea añadir otra restricción
        otra_mas = input("\n¿Deseas añadir otra restricción? (s/n): ").lower()
        if otra_mas != 's':
            break
        i += 1
        
    print("\n¡Problema ingresado con éxito!")
    return tipo_problema, funcion_objetivo, restricciones