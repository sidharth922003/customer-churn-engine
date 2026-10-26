import csv
import random
import math
import os

def generate_customer_dataset(num_records=10000, output_file="data/customer_churn_data.csv"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    random.seed(42)

    headers = [
        "customer_id",
        "tenure_months",
        "contract_type",
        "monthly_charges",
        "total_charges",
        "support_tickets_90d",
        "failed_transactions_90d",
        "avg_daily_usage_mins",
        "days_since_last_login",
        "has_auto_renew",
        "churn"
    ]

    records = []
    contract_options = ["Month-to-month", "One year", "Two year"]

    for i in range(1, num_records + 1):
        cid = f"CUST-{100000 + i}"
        
        contract = random.choices(contract_options, weights=[0.55, 0.25, 0.20])[0]
        
        if contract == "Month-to-month":
            tenure = random.randint(1, 36)
        elif contract == "One year":
            tenure = random.randint(12, 48)
        else:
            tenure = random.randint(24, 72)

        monthly = round(random.uniform(25.0, 120.0), 2)
        total = round(monthly * tenure * random.uniform(0.95, 1.05), 2)

        support_tickets = random.choices([0, 1, 2, 3, 4, 5, 7], weights=[0.4, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02])[0]
        failed_tx = random.choices([0, 1, 2, 3], weights=[0.75, 0.15, 0.07, 0.03])[0]
        usage_mins = round(random.uniform(10.0, 180.0), 1)
        days_inactive = random.randint(0, 35)
        auto_renew = 1 if contract != "Month-to-month" else random.choice([0, 1])

        # Churn risk scoring formula (generates realistic ground truth ~15% class imbalance)
        risk_score = 0.0
        
        if contract == "Month-to-month":
            risk_score += 0.35
        elif contract == "One year":
            risk_score -= 0.15
        else:
            risk_score -= 0.35

        if tenure < 6:
            risk_score += 0.25
        elif tenure > 36:
            risk_score -= 0.20

        if support_tickets >= 3:
            risk_score += 0.30 * (support_tickets / 3.0)
        
        if failed_tx >= 1:
            risk_score += 0.25 * failed_tx

        if days_inactive > 14:
            risk_score += 0.25 * (days_inactive / 14.0)

        if usage_mins < 30.0:
            risk_score += 0.20
        elif usage_mins > 90.0:
            risk_score -= 0.15

        if auto_renew == 0:
            risk_score += 0.15

        # Sigmoid probability with realistic baseline (~18-20% churn rate)
        prob = 1.0 / (1.0 + math.exp(-2.5 * (risk_score - 1.1)))
        churn = 1 if (random.random() < prob) else 0

        records.append([
            cid, tenure, contract, monthly, total,
            support_tickets, failed_tx, usage_mins,
            days_inactive, auto_renew, churn
        ])

    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(records)

    churn_count = sum(r[-1] for r in records)
    print(f"Generated {num_records} records in '{output_file}'.")
    print(f"Churn rate: {churn_count}/{num_records} ({churn_count / num_records * 100:.2f}%)")

if __name__ == "__main__":
    generate_customer_dataset()
