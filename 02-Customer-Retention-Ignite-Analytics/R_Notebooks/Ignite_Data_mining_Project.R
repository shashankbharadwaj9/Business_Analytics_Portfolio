# Load dataset
customer_data <- read.csv("C:\\Users\\ashas\\Downloads\\customer_data.csv")



# Load required libraries
library(tidyverse)
library(cluster)

# Explore the data
summary(customer_data)
str(customer_data)
head(customer_data)

# Check for missing values
missing_values <- colSums(is.na(customer_data))
print(missing_values)

# Select relevant features for clustering
cluster_data <- customer_data[, c("age", "salary", "spending_score")]

# Visualizing distribution of selected features, This Helps to find out potential outliers in the data.
ggplot(customer_data, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "blue", color = "black") +
  labs(title = "Distribution of Age", x = "Age", y = "Frequency") +
  theme_minimal()  

ggplot(customer_data, aes(x = salary)) +
  geom_histogram(binwidth = 5000, fill = "salmon", color = "black") +
  labs(title = "Distribution of Salary", x = "Salary", y = "Frequency") +
  theme_minimal()

ggplot(customer_data, aes(x = spending_score)) +
  geom_histogram(binwidth = 5000, fill = "salmon", color = "black") +
  labs(title = "Distribution of Salary", x = "Salary", y = "Frequency") +
  theme_minimal()

# Scale the data
# Scaling can improve the performance of clustering algorithms by ensuring all features are on a similar scale.
scaled_data <- scale(cluster_data)

#selecting K value
# Using ELBLOW to find optimal number of clusters
set.seed(6)
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(scaled_data, i)$withinss)
plot(x = 1:10,
     y = wcss,
     type = 'b',
     main = paste('The Elbow Method'),
     xlab = 'Number of clusters',
     ylab = 'WCSS')

# Choose the optimal number of clusters based on the silhouette plot
k <- 2

# Perform k-means clustering
km_model <- kmeans(scaled_data, centers = k, iter.max = 600,
                   nstart = 10)

# Add cluster labels to the dataset
# This adds new column to the customer dataset.
customer_data$cluster <- as.factor(km_model$cluster)

# Visualize clusters
library(cluster)
clusplot(x = cluster_data,
         clus = km_model$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of customers'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')

#Loading data
purchases <- read.csv("C:\\Users\\ashas\\Downloads\\customer_purchase_history_final.csv")

dataset = read.transactions("C:\\Users\\ashas\\Downloads\\customer_purchase_history_final.csv", sep = ',', rm.duplicates = TRUE)

library(arules)
library(arulesViz)

# Data preprocessing and exploration
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)


# Training Apriori on the dataset
rules <- apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Visualising the results
inspect(sort(rules, by = 'lift')[1:10])

# Training Eclat on the dataset
rules <- eclat(data = dataset, parameter = list(support = 0.003, minlen = 2))

# Visualising the results
inspect(sort(rules, by = 'support')[1:10])

