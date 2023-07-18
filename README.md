# CLTV_Prediction_FLO

![Fotoğraf Açıklaması](https://github.com/hgatasagun/CLTV_Prediction_FLO/raw/main/Customer.jpg)


This project aims to predict the potential value that customers will bring to FLO, a company engaged in sales and marketing activities. By estimating Customer Lifetime Value (CLTV), FLO can make informed decisions and develop a roadmap for its sales and marketing strategies.

## Dataset Story

The dataset used for this project contains historical data of customers who made purchases from FLO as OmniChannel customers (both online and offline) between 2020 and 2021. It provides insights into their shopping behavior, including the channels used, order dates, total order counts, total customer value, and categories of interest. This data serves as the foundation for analyzing customer patterns and estimating their future value to the company.

## Variables

- master_id: Unique customer identifier
- order_channel: Channel used for the purchase (Android, iOS, Desktop, Mobile)
- last_order_channel: Channel used for the latest purchase
- first_order_date: Date of the first purchase
- last_order_date: Date of the latest purchase
- last_order_date_online: Date of the latest online purchase
- last_order_date_offline: Date of the latest offline purchase
- order_num_total_ever_online: Total number of purchases made online
- order_num_total_ever_offline: Total number of purchases made offline
- customer_value_total_ever_offline: Total amount paid in offline purchases
- customer_value_total_ever_online: Total amount paid in online purchases
- interested_in_categories_12: List of categories in which the customer made purchases in the last 12 months
