-- Each query starts with a unique name and is executed by `credit-risk db-analysis`.

-- name: default_rate
SELECT
    COUNT(*) AS client_count,
    SUM(default_next_month)::BIGINT AS defaults,
    AVG(default_next_month::NUMERIC) AS default_rate
FROM targets;

-- name: limit_age_summary
SELECT
    COUNT(*) AS client_count,
    AVG(c.limit_bal)::NUMERIC(14, 2) AS mean_credit_limit,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY c.limit_bal) AS median_credit_limit,
    AVG(c.age)::NUMERIC(8, 2) AS mean_age,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY c.age) AS median_age
FROM clients AS c;

-- name: monthly_amount_summary
SELECT
    month_index,
    COUNT(*) AS client_month_rows,
    AVG(bill_amount)::NUMERIC(16, 2) AS mean_bill_amount,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY bill_amount) AS median_bill_amount,
    AVG(payment_amount)::NUMERIC(16, 2) AS mean_payment_amount,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY payment_amount) AS median_payment_amount
FROM monthly_history
GROUP BY month_index
ORDER BY month_index;

-- name: repayment_status_distribution
SELECT
    month_index,
    repayment_status,
    COUNT(*) AS client_count,
    COUNT(*) FILTER (WHERE repayment_status > 0) AS delayed_count
FROM monthly_history
GROUP BY month_index, repayment_status
ORDER BY month_index, repayment_status;

-- name: clients_with_repeated_delays
WITH client_delay_counts AS (
    SELECT
        client_id,
        COUNT(*) FILTER (WHERE repayment_status > 0) AS delayed_month_count
    FROM monthly_history
    GROUP BY client_id
    HAVING COUNT(*) FILTER (WHERE repayment_status > 0) >= 2
)
SELECT delayed_month_count, COUNT(*) AS client_count
FROM client_delay_counts
GROUP BY delayed_month_count
ORDER BY delayed_month_count;

-- name: monthly_bill_change
WITH bill_history_with_previous AS (
    SELECT
        client_id,
        month_index,
        bill_amount,
        LAG(bill_amount) OVER (
            PARTITION BY client_id ORDER BY month_index
        ) AS previous_bill_amount
    FROM monthly_history
)
SELECT
    month_index,
    COUNT(*) FILTER (WHERE previous_bill_amount IS NOT NULL) AS client_transitions,
    AVG(bill_amount - previous_bill_amount)::NUMERIC(16, 2) AS mean_bill_change
FROM bill_history_with_previous
GROUP BY month_index
ORDER BY month_index;

-- name: top_clients_by_total_payment
WITH client_payment_totals AS (
    SELECT client_id, SUM(payment_amount)::BIGINT AS total_payment_amount
    FROM monthly_history
    GROUP BY client_id
), ranked_clients AS (
    SELECT
        client_id,
        total_payment_amount,
        ROW_NUMBER() OVER (
            ORDER BY total_payment_amount DESC, client_id
        ) AS payment_rank
    FROM client_payment_totals
)
SELECT client_id, total_payment_amount, payment_rank
FROM ranked_clients
WHERE payment_rank <= 10
ORDER BY payment_rank;

-- name: join_integrity
SELECT
    (SELECT COUNT(*) FROM clients) AS clients,
    (SELECT COUNT(*) FROM targets) AS targets,
    (SELECT COUNT(DISTINCT client_id) FROM monthly_history) AS history_clients,
    (SELECT COUNT(*) FROM clients c JOIN targets t USING (client_id)) AS clients_with_target,
    (SELECT COUNT(*) FROM ml_feature_dataset) AS feature_rows;
