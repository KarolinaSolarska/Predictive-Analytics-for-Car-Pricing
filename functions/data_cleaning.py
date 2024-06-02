import pandas as pd

def prepare_data(data):
    data = data[data['Kierownica po prawej (Anglik)'] != 'Tak']

    columns_to_keep = [
    'Marka pojazdu', 'Model pojazdu', 'Rok produkcji', 'Przebieg', 'Pojemność skokowa', 'Rodzaj paliwa', 
    'Moc', 'Skrzynia biegów', 'Napęd', 'Typ nadwozia', 'Liczba drzwi', 'Kolor', 'Metalik', 
    'Kraj pochodzenia', 'Pierwszy właściciel', 'Bezwypadkowy', 'Stan', 'Cena', 'Waluta']

    data = data[columns_to_keep]

    # Dictionary to map Polish column names to English column names
    column_mapping = {
        'Marka pojazdu': 'brand',
        'Model pojazdu': 'model',
        'Rok produkcji': 'year_production',    
        'Przebieg': 'mileage',
        'Pojemność skokowa': 'engine_capacity',
        'Rodzaj paliwa': 'fuel_type',
        'Moc': 'power',
        'Skrzynia biegów': 'gearbox',
        'Napęd': 'drive_type',
        'Typ nadwozia': 'body_type',
        'Liczba drzwi': 'doors',
        'Kolor': 'color',
        'Metalik': 'metallic',
        'Kraj pochodzenia': 'country_origin',
        'Pierwszy właściciel': 'first_owner',
        'Bezwypadkowy': 'accident_free',
        'Stan': 'condition',
        'Cena': 'price',
        'Waluta': 'currency'
    }

    # Rename the columns
    data = data.rename(columns=column_mapping)

# def clean_data(data):


