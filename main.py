# archivo: main.py

from entrada_usuario import obtener_datos_lp
import metodo_simplex
# ¡Importamos el nuevo módulo gráfico!
import metodo_grafico

def menu_principal():
    """
    Muestra el menú principal y dirige el flujo de la aplicación.
    """
    tipo_problema, funcion_objetivo, restricciones = obtener_datos_lp()
    
    print("\n--- Resumen del Problema Ingresado ---")
    # ... (el código de resumen del problema no cambia) ...
    print(f"Tipo de Problema: {'Maximizar' if tipo_problema == 'max' else 'Minimizar'}")
    z_str = " + ".join([f"{c}x{i+1}" for i, c in enumerate(funcion_objetivo)])
    print(f"Función Objetivo: {z_str}")
    print("Sujeto a las siguientes restricciones:")
    for i, r in enumerate(restricciones):
        r_str = " + ".join([f"{c}x{j+1}" for j, c in enumerate(r['coeficientes'])])
        print(f"  {i+1}) {r_str} {r['desigualdad']} {r['rhs']}")

    print("\n--- ¿Qué método deseas utilizar? ---")
    print("1. Método Simplex")
    print("2. Método Gráfico (solo si el problema tiene 2 variables)")
    
    while True:
        opcion = input("Selecciona una opción (1 o 2): ")
        if opcion == '1':
            print("\nIniciando solución con Método Simplex...")
            metodo_simplex.resolver_simplex(tipo_problema, funcion_objetivo, restricciones)
            break
        elif opcion == '2':
            if len(funcion_objetivo) == 2:
                print("\nIniciando solución con Método Gráfico...")
                # --- MODIFICACIÓN AQUÍ ---
                metodo_grafico.resolver_grafico(tipo_problema, funcion_objetivo, restricciones)
                break
            else:
                print("Error: El método gráfico solo se puede usar con 2 variables.")
        else:
            print("Opción no válida. Inténtalo de nuevo.")

if __name__ == "__main__":
    menu_principal()