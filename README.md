Problem Statement:

Imagine we work for a subscription company such as a telecom provider.
They have thousands of customers.
Some customers leave every month.

The company wants to identify customers who are likely to leave before they actually leave.

For example:

| Customer | Tenure | Monthly Charges | Support Calls | Contract | Churn |
| -------- | -----: | --------------: | ------------: | -------- | ----- |
| A        |      2 |              85 |             6 | Monthly  | Yes   |
| B        |     48 |              45 |             1 | Two Year | No    |
| C        |      5 |              90 |             5 | Monthly  | Yes   |
| D        |     36 |              55 |             1 | One Year | No    |

Our model should eventually do this:
New Customer
     ↓
ML Model
     ↓
Probability of Churn
     ↓
0.87
     ↓
Likely to Churn

The business could then contact that customer with a retention offer.
