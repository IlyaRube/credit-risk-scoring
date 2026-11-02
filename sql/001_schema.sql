-- Normalized relational schema for UCI Default of Credit Card Clients.
CREATE TABLE IF NOT EXISTS clients (
    client_id BIGINT PRIMARY KEY CHECK (client_id > 0),
    limit_bal BIGINT NOT NULL,
    sex SMALLINT NOT NULL,
    education SMALLINT NOT NULL,
    marriage SMALLINT NOT NULL,
    age SMALLINT NOT NULL CHECK (age > 0)
);

CREATE TABLE IF NOT EXISTS monthly_history (
    client_id BIGINT NOT NULL REFERENCES clients (client_id) ON DELETE CASCADE,
    month_index SMALLINT NOT NULL CHECK (month_index BETWEEN 1 AND 6),
    repayment_status SMALLINT NOT NULL,
    bill_amount BIGINT NOT NULL,
    payment_amount BIGINT NOT NULL,
    CONSTRAINT pk_monthly_history PRIMARY KEY (client_id, month_index)
);

CREATE TABLE IF NOT EXISTS targets (
    client_id BIGINT PRIMARY KEY REFERENCES clients (client_id) ON DELETE CASCADE,
    default_next_month SMALLINT NOT NULL CHECK (default_next_month IN (0, 1))
);

CREATE INDEX IF NOT EXISTS idx_monthly_history_month_status
    ON monthly_history (month_index, repayment_status);
