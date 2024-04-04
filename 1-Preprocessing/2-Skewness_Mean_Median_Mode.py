import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

def ShowInformationDataFrame(df, name):
    print(f"{name}:")
    print(df.head())
    print(df.describe())
    print("")

def main():
    input_file = '0-Datasets/DatasetJobsScienceDadosClear.csv'
    

    df = pd.read_csv(input_file, header=0)  
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Separando os recursos e o alvo
    features = numeric_columns[:-1].tolist()
    target = [numeric_columns[-1]]

    ShowInformationDataFrame(df, "Dataframe Original")

    x = df.loc[:, features].values
    y = df.loc[:, target].values

    x_zscore = StandardScaler().fit_transform(x)
    normalized1_df = pd.DataFrame(data=x_zscore, columns=features)
    normalized1_df = pd.concat([normalized1_df, df[target]], axis=1)
    ShowInformationDataFrame(normalized1_df, "Dataframe Z-Score Normalized")

    # Min-Max normalization
    x_minmax = MinMaxScaler().fit_transform(x)
    normalized2_df = pd.DataFrame(data=x_minmax, columns=features)
    normalized2_df = pd.concat([normalized2_df, df[target]], axis=1)
    ShowInformationDataFrame(normalized2_df, "Dataframe Min-Max Normalized")

    # Robust Scaler
    x_robust = RobustScaler().fit_transform(x)
    normalized3_df = pd.DataFrame(data=x_robust, columns=features)
    normalized3_df = pd.concat([normalized3_df, df[target]], axis=1)
    ShowInformationDataFrame(normalized3_df, "Dataframe Robust Normalized")

    # Unit Vector Transformation
    x_norma = Normalizer().fit_transform(x)
    normalized4_df = pd.DataFrame(data=x_norma, columns=features)
    normalized4_df = pd.concat([normalized4_df, df[target]], axis=1)
    ShowInformationDataFrame(normalized4_df, "Dataframe Unit Vector Normalized")
    
    # Loop para salvar cada dataframe separadamente
    for method, normalized_df in [('Z-Score', normalized1_df), ('Min-Max', normalized2_df),
                                ('Robust', normalized3_df), ('Unit-Vector', normalized3_df)]:
        output_file = f'0-Datasets/Normatização_{method}_{input_file.split("/")[-1]}'
        normalized_df.to_csv(output_file, header=True, index=False)

if __name__ == "__main__":
    main()
