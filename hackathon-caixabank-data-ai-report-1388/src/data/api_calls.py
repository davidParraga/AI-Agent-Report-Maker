# file for implementing api call functions
import requests

def get_client_data(client_id):
    url = f"https://faas-lon1-917a94a7.doserverless.co/api/v1/web/fn-a1f52b59-3551-477f-b8f3-de612fbf2769/default/clients-data?client_id={client_id}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error al obtener datos del cliente {client_id}: {e}")
        return None

def get_card_data(client_id):
    url = f"https://faas-lon1-917a94a7.doserverless.co/api/v1/web/fn-a1f52b59-3551-477f-b8f3-de612fbf2769/default/cards-data?client_id={client_id}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error al obtener datos de tarjetas del cliente {client_id}: {e}")
        return None
