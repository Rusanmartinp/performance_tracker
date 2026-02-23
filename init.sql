CREATE TABLE IF NOT EXISTS products (
    id       SERIAL PRIMARY KEY,
    name     TEXT NOT NULL,
    price    NUMERIC(10,2),
    category TEXT
);

CREATE TABLE IF NOT EXISTS daily_performance (
    date        DATE,
    product_id  INTEGER REFERENCES products(id),
    impressions INTEGER,
    clicks      INTEGER,
    ad_spend    NUMERIC(10,2),
    units_sold  INTEGER,
    revenue     NUMERIC(10,2),
    PRIMARY KEY (date, product_id)
);
