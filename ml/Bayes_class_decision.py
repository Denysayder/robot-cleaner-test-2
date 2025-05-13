import pandas as pd
import numpy as np

def save_parameters_to_csv(parameter_file_path, mean, std):
    # Tạo DataFrame chứa thông tin về các tham số
    parameters_df = pd.DataFrame({
        'Parameter': ['mean', 'std'],
        'Values': [mean, std]
    })
    # Xuất DataFrame vào file CSV
    parameters_df.to_csv(parameter_file_path, index=False)

def estimate_gaussian_parameters(file_path, Parameter_file_path, feature_name='Values'):
    # Đọc dữ liệu từ file CSV
    data = pd.read_csv(file_path)
    # Trích xuất đặc trưng từ dữ liệu
    feature_values = data[feature_name].values
    # Ước lượng tham số cho phân phối Gaussian (mean và standard deviation)
    mean = np.mean(feature_values)
    std = np.std(feature_values)
    save_parameters_to_csv(Parameter_file_path, mean, std)

def get_parameters(parameter_file_path):
    # Đọc dữ liệu từ file CSV
    data = pd.read_csv(parameter_file_path)
    # Lấy giá trị mean và std từ DataFrame
    mean = data[data['Parameter'] == 'mean']['Values'].values[0]
    std = data[data['Parameter'] == 'std']['Values'].values[0]
    return mean, std

# Dự đoán nhóm của số mới
def predict_group(value, mean_clean, std_clean, mean_dirty, std_dirty):
    p_clean = 1 / (np.sqrt(2 * np.pi) * std_clean) * np.exp(-0.5 * ((value - mean_clean) / std_clean) ** 2)
    p_dirty = 1 / (np.sqrt(2 * np.pi) * std_dirty) * np.exp(-0.5 * ((value - mean_dirty) / std_dirty) ** 2)
    if p_clean > p_dirty:
        return "Clean"
    else:
        return "G"

if __name__ == '__main__':
    new_value = 0.67
    # Sử dụng hàm predict_group để dự đoán nhóm của số mới
    clean_file = 'data/processed/data_100_150.csv'
    dirty_file = 'data/processed/data_100_150_dirty.csv'
    clean_parameters_file = 'data/raw/clean_parameters.csv'
    dirty_parameters_file = 'data/raw/dirty_parameters.csv'

    estimate_gaussian_parameters(clean_file, clean_parameters_file)
    estimate_gaussian_parameters(dirty_file, dirty_parameters_file)

    # Get the parameters for the Bayes Classifier
    mean_clean, std_clean = get_parameters('data/raw/clean_parameters.csv')
    mean_dirty, std_dirty = get_parameters('data/raw/dirty_parameters.csv')

    predicted_group = predict_group(new_value, mean_clean, std_clean, mean_dirty, std_dirty)
    print("Number {} is belong to group : {}".format(new_value, predicted_group))
