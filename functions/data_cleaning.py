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

    return data

def clean_data(data):

    # drop columns
    cols_to_delete = ['model', 'metallic', 'country_origin', 'first_owner']
    data = data.drop(columns=cols_to_delete)

    # drop rows with missing values
    cols_to_drop_na = data.columns[~data.columns.isin(['accident_free'])]
    data = data.dropna(subset=cols_to_drop_na)

    ### brand
    unique_brand = data['brand'].value_counts()
    # Identify brands with counts >= 100 - maybe define by brand - to have always the same brands
    brands_to_keep = unique_brand[unique_brand >= 100].index
    # Filter the dataset to keep only these brands
    data = data[data['brand'].isin(brands_to_keep)]

    # Perform one-hot encoding for the 'brand' column
    data = pd.get_dummies(data, columns=['brand'], prefix='', prefix_sep='')


    ### year_production
    # Convert 'year_production' to integer
    data['year_production'] = data['year_production'].astype(int)


    ### mileage
    # Remove ' km' from the strings and any spaces, then convert to integer
    data['mileage'] = data['mileage'].str.replace(' km', '').str.replace(' ', '').astype(int)

    # Remove extreme outliers 
    data = data[(data['mileage'] < 800000)]


    ### engine_capacity
    # Remove ' km' from the strings and any spaces, then convert to integer
    data['engine_capacity'] = data['engine_capacity'].str.replace(' cm3', '').str.replace(' ', '').astype(int)


    ### fuel_type
    # Remove outliers
    data = data[~data['fuel_type'].isin(['Benzyna+CNG', 'Wodór'])]
    # Perform one-hot encoding
    data = pd.get_dummies(data, columns=['fuel_type'])

    # Remove the 'fuel_type_' prefix from the column names
    data.columns = data.columns.str.replace('fuel_type_', '')


    ### power
    # Remove ' km' from the strings and any spaces, then convert to integer
    data['power'] = data['power'].str.replace(' KM', '').str.replace(' ', '').astype(int)

    # Remove extreme outliers 
    data = data[(data['power'] < 1501) & (data['power'] > 4)]


    ### gearbox
    # Map the 'gearbox' column to binary values
    data['gearbox'] = data['gearbox'].map({'Manualna': 0, 'Automatyczna': 1})
    # Rename the column
    data = data.rename(columns={'gearbox': 'automatic_gearbox'})


    ### drive_type
    data['drive_type'] = data['drive_type'].replace({
        'Na przednie koła': 'Front_wheel_drive',
        'Na tylne koła': 'Rear_wheel_drive',
        '4x4 (stały)': '4x4',
        '4x4 (dołączany automatycznie)': '4x4',
        '4x4 (dołączany ręcznie)': '4x4'
    })

    # Perform one-hot encoding
    data = pd.get_dummies(data, columns=['drive_type'])

    # Remove the prefix from the column names
    data.columns = data.columns.str.replace('drive_type_', '')


    ### body_type
    body_type_mapping = {
        'Kombi': 'Combi',
        'Kompakt': 'Compact',
        'Auta miejskie': 'City_cars',
        'Auta małe': 'Small_cars',
        'Kabriolet': 'Cabriolet'
    }
    # First, translate the categories to English
    data['body_type'] = data['body_type'].replace(body_type_mapping)
    # delete observations where body type not in the mapping dictionary
    data = data[data['body_type'].isin(body_type_mapping.values())]
    # Perform one-hot encoding
    data = pd.get_dummies(data, columns=['body_type'])

    # Remove the prefix from the column names
    data.columns = data.columns.str.replace('body_type_', '')


    ### doors
    # Convert to integer
    data['doors'] = data['doors'].astype(int)
    # remove values > 5
    data = data[data['doors'] <= 5]
    # change the number of doors to binary values
    data['doors'] = data['doors'].replace({
        2: 0,
        3: 0,
        4: 1,
        5: 1
    })

    # rename the doors column
    data = data.rename(columns={'doors': 'doors_5'})

    ### color
    # define a mapping dictionary for color
    color_mapping = {
        'Czarny': 'Black',
        'Szary': 'Gray',
        'Biały': 'White',
        'Srebrny': 'Silver',
        'Niebieski': 'Blue',
        'Czerwony': 'Red',
        'Inny kolor': 'Other_color',
        'Granatowy': 'Navy_blue',
        'Brązowy': 'Brown',
        'Zielony': 'Green',
        'Bordowy': 'Burgundy',
        'Beżowy': 'Beige',
        'Złoty': 'Gold',
        'Błękitny': 'Light_blue',
        'Pomarańczowy': 'Orange',
        'Żółty': 'Yellow',
        'Fioletowy': 'Purple'
    }

    # remove observations when color is not in the mapping dictionary
    data = data[data['color'].isin(color_mapping.keys())]
    # First, translate the categories to English
    data['color'] = data['color'].replace(color_mapping)
    # Perform one-hot encoding 
    data = pd.get_dummies(data, columns=['color'])
    # Remove the prefix from the column names
    data.columns = data.columns.str.replace('color_', '')

    ### accident_free
    # Replace 'Tak' with 1 and fill missing values with 0
    data['accident_free'] = data['accident_free'].replace({'Tak': 1})
    data['accident_free'] = data['accident_free'].fillna(0)

    # change the type to int
    data['accident_free'] = data['accident_free'].astype(int)


    ### condition
    condition_mapping = {
        'Używane': 'Used_cars',
        'Nowe': 'New_cars',
        'Używany': 'Used_cars',
        'Nowy': 'New_cars'
    }
    # remove observations when condition is not in the mapping dictionary
    data = data[data['condition'].isin(condition_mapping.keys())]
    # First, translate the categories to English
    data['condition'] = data['condition'].replace(condition_mapping)

    # Perform one-hot encoding
    data = pd.get_dummies(data, columns=['condition'])

    # Remove the prefix from the column names
    data.columns = data.columns.str.replace('condition_', '')

    ### price
    # change data type as numeric
    data['price'] = pd.to_numeric(data['price'], errors='coerce')
    # drop rows with missing values which arise after conversion
    data = data.dropna(subset=['price'])

    ### currency
    # delete observations where currency not in PLN or EUR
    data = data[data['currency'].isin(['PLN', 'EUR'])]
    # Define the exchange rate
    eur_to_pln_rate = 4.2077
    # Convert prices in EUR to PLN
    data.loc[data['currency'] == 'EUR', 'price'] = data.loc[data['currency'] == 'EUR', 'price'] * eur_to_pln_rate
    # Update the currency column to PLN
    data['currency'] = 'PLN'
    # Convert to integer
    data['price'] = data['price'].astype(int)

    return data
