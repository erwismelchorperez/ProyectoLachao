import pandas as pd
import numpy as np
import os

def ProcesarFile(file, anio, mes, flag):
    namearchivo = mes.split(".")[0].lower()
    if flag:
        archivo = pd.read_excel(file, sheet_name=0)
    else:
        archivo = pd.read_excel(file, sheet_name=0, header=1)
        archivo = archivo[archivo["Sucursal"] != "No. de Pagina: 1"]
    archivo["anio"] = anio
    archivo["mes"] = namearchivo[0:3]

    return archivo

def ProcesarFileCartera(file, anio, mes, flag):
    namearchivo = mes.split(".")[0].lower()
    archivo = pd.read_excel(file, skiprows=5, header=0)
    #print(archivo)
    # Eliminar donde "Nombre" contenga "Ana" o "Pedro"
    archivo = archivo[~archivo["Nombre del Acreditado"].str.contains("SubTotal|Naturaleza:|Nombre del|Producto:|Sucursal:|CREDITOS|Total Cartera|Tipo de Crédito:", case=False, na=False)]
    archivo["anio"] = anio
    archivo["mes"] = namearchivo[0:3]

    return archivo

def ProcesarCarpetas(path, tipoarchivo, flag):
    df_total = pd.DataFrame()
    for elemento in os.listdir(path):
        ruta_completa = os.path.join(path, elemento)
        #print(ruta_completa)
        for file in os.listdir(ruta_completa):
            print("Archivo:", ruta_completa+"/"+file, "   ", elemento, "            ", file)
            # Apilar verticalmente
            if tipoarchivo == "Cartera":
                archivo = ProcesarFileCartera(ruta_completa+"/"+file, anio =elemento, mes=file, flag= flag)
            else:
                archivo = ProcesarFile(ruta_completa+"/"+file, anio =elemento, mes=file, flag= flag)
            if df_total.empty:
                df_total = archivo
            else:
                df_total = pd.concat([df_total, archivo], ignore_index=True)
            #break
            print(df_total.shape)
    df_total.to_csv(tipoarchivo+".csv", index=False)

if __name__ == "__main__":
    #path = "./../BaseLachao/CAPITAL SOCIAL"
    #ProcesarCarpetas(path,"Socios", False)
    path = "./../BaseLachao/CARTERA"
    ProcesarCarpetas(path,"Cartera", True)
    path = "./../BaseLachao/CAPTACIÓN"
    ProcesarCarpetas(path,"Captacion", True)
