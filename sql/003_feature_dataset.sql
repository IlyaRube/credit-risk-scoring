-- This view contains predictors and the client key only. The target is queried separately.
CREATE OR REPLACE VIEW ml_feature_dataset AS
WITH history_pivot AS (
    SELECT
        client_id,
        MAX(CASE WHEN month_index = 1 THEN repayment_status END) AS repayment_status_m1,
        MAX(CASE WHEN month_index = 2 THEN repayment_status END) AS repayment_status_m2,
        MAX(CASE WHEN month_index = 3 THEN repayment_status END) AS repayment_status_m3,
        MAX(CASE WHEN month_index = 4 THEN repayment_status END) AS repayment_status_m4,
        MAX(CASE WHEN month_index = 5 THEN repayment_status END) AS repayment_status_m5,
        MAX(CASE WHEN month_index = 6 THEN repayment_status END) AS repayment_status_m6,
        MAX(CASE WHEN month_index = 1 THEN bill_amount END) AS bill_amount_m1,
        MAX(CASE WHEN month_index = 2 THEN bill_amount END) AS bill_amount_m2,
        MAX(CASE WHEN month_index = 3 THEN bill_amount END) AS bill_amount_m3,
        MAX(CASE WHEN month_index = 4 THEN bill_amount END) AS bill_amount_m4,
        MAX(CASE WHEN month_index = 5 THEN bill_amount END) AS bill_amount_m5,
        MAX(CASE WHEN month_index = 6 THEN bill_amount END) AS bill_amount_m6,
        MAX(CASE WHEN month_index = 1 THEN payment_amount END) AS payment_amount_m1,
        MAX(CASE WHEN month_index = 2 THEN payment_amount END) AS payment_amount_m2,
        MAX(CASE WHEN month_index = 3 THEN payment_amount END) AS payment_amount_m3,
        MAX(CASE WHEN month_index = 4 THEN payment_amount END) AS payment_amount_m4,
        MAX(CASE WHEN month_index = 5 THEN payment_amount END) AS payment_amount_m5,
        MAX(CASE WHEN month_index = 6 THEN payment_amount END) AS payment_amount_m6
    FROM monthly_history
    GROUP BY client_id
),
history_aggregates AS (
    SELECT
        client_id,
        MAX(repayment_status) AS max_repayment_status,
        COUNT(*) FILTER (WHERE repayment_status > 0) AS delinquent_month_count,
        AVG(bill_amount)::DOUBLE PRECISION AS avg_bill_amount,
        AVG(payment_amount)::DOUBLE PRECISION AS avg_payment_amount,
        SUM(payment_amount)::BIGINT AS total_payment_amount,
        (
            MAX(CASE WHEN month_index = 6 THEN bill_amount END)
            - MAX(CASE WHEN month_index = 1 THEN bill_amount END)
        )::BIGINT AS bill_amount_change_6m,
        (
            SUM(payment_amount)::NUMERIC
            / NULLIF(SUM(ABS(bill_amount)), 0)
        )::DOUBLE PRECISION AS payment_to_abs_bill_ratio
    FROM monthly_history
    GROUP BY client_id
)
SELECT
    c.client_id,
    c.limit_bal,
    c.sex,
    c.education,
    c.marriage,
    c.age,
    p.repayment_status_m1,
    p.repayment_status_m2,
    p.repayment_status_m3,
    p.repayment_status_m4,
    p.repayment_status_m5,
    p.repayment_status_m6,
    p.bill_amount_m1,
    p.bill_amount_m2,
    p.bill_amount_m3,
    p.bill_amount_m4,
    p.bill_amount_m5,
    p.bill_amount_m6,
    p.payment_amount_m1,
    p.payment_amount_m2,
    p.payment_amount_m3,
    p.payment_amount_m4,
    p.payment_amount_m5,
    p.payment_amount_m6,
    a.max_repayment_status,
    a.delinquent_month_count,
    a.avg_bill_amount,
    a.avg_payment_amount,
    a.total_payment_amount,
    a.bill_amount_change_6m,
    a.payment_to_abs_bill_ratio
FROM clients AS c
JOIN history_pivot AS p USING (client_id)
JOIN history_aggregates AS a USING (client_id);
