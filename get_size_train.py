# Phân tích khám phá dữ liệu thông qua thống kê và các biểu đồ
# Chỉ được phân tích trên tập huấn luyện

# 1. Đọc dữ liệu
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# --- Yêu cầu 1: Phân tích khám phá dữ liệu ---

# a. In ra số dòng, số cột, tên cột, kiểu dữ liệu [cite: 24]
print(f"Kích thước tập huấn luyện: {train_df.shape}")
print("-" * 30)
print(train_df.info())

# b. Kiểm tra số lượng giá trị thiếu theo từng cột [cite: 25]
missing_values = train_df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
print("\nCác cột có giá trị thiếu:")
print(missing_values)

# Trực quan hóa giá trị thiếu (tùy chọn để báo cáo đẹp hơn)
plt.figure(figsize=(12, 6))
sns.barplot(x=missing_values.index, y=missing_values.values)
plt.xticks(rotation=90)
plt.title("Thống kê giá trị thiếu trên tập huấn luyện")
plt.show()

# c. Thống kê mô tả cho các đặc trưng số [cite: 26]
# Chỉ lấy các cột có kiểu dữ liệu là số
numeric_features = train_df.select_dtypes(include=[np.number])
print("\nThống kê mô tả cho các đặc trưng số:")
print(numeric_features.describe())

# d. Phân tích phân phối của biến mục tiêu SalePrice [cite: 27]
plt.figure(figsize=(10, 6))
sns.histplot(train_df['SalePrice'], kde=True, color='blue')
plt.title("Phân phối của giá nhà (SalePrice)")
plt.xlabel("Giá bán")
plt.ylabel("Tần suất")
plt.show()

print(f"\nĐộ lệch (Skewness) của SalePrice: {train_df['SalePrice'].skew():.2f}")

# e. Vẽ các biểu đồ để hiểu rõ dữ liệu [cite: 28]

# Heatmap: Ma trận tương quan để tìm 5 đặc trưng tốt nhất cho câu 2b 
plt.figure(figsize=(12, 10))
correlations = numeric_features.corr()
# Lấy top 10 đặc trưng tương quan nhất với SalePrice
top_corr_features = correlations['SalePrice'].sort_values(ascending=False).head(11).index
sns.heatmap(train_df[top_corr_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Ma trận tương quan giữa các đặc trưng hàng đầu với SalePrice")
plt.show()

# Scatter plot: Ví dụ diện tích sống (GrLivArea) vs SalePrice
plt.figure(figsize=(8, 6))
sns.scatterplot(data=train_df, x='GrLivArea', y='SalePrice')
plt.title("Mối quan hệ giữa Diện tích sống và Giá nhà")
plt.show()

# Boxplot: Ví dụ Chất lượng tổng thể (OverallQual) vs SalePrice
plt.figure(figsize=(10, 6))
sns.boxplot(data=train_df, x='OverallQual', y='SalePrice')
plt.title("Giá nhà theo chất lượng tổng thể")
plt.show()