# CLTV_Prediction_FLO

This project aims to predict the potential value that customers will bring to FLO, based on their past shopping behavior as OmniChannel (both online and offline) customers between 2020 and 2021.

## Dataset Story

The dataset consists of historical shopping behavior data of customers who made purchases from FLO through multiple channels. It includes information about their order channels, order dates, total order counts, total customer value, and categories of interest.

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

These variables capture customers' shopping habits, channel preferences, spending patterns, and categories of interest. They play a crucial role in building CLTV prediction models and segmenting customers based on their behavior and value to the company.
