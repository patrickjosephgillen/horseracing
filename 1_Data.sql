-- This script takes about 7.5 minutes to execute

/*
 * Create my_races
 */

DROP TABLE IF EXISTS my_races;

CREATE TABLE my_races AS SELECT b.race_id,
    COUNT(runner_id) AS runners,
    meeting_date,
    course,
    going,
    direction,
    class,
    handicap,
    seller,
    claimer,
    apprentice,
    maiden,
    amateur,
    rating,
    group_race,
    min_age,
    max_age,
    distance_yards,
    added_money FROM
    historic_runners a
        INNER JOIN
    historic_races b
WHERE
    a.race_id = b.race_id
        AND all_weather = 1
        AND race_type = 'All Weather Flat'
        AND course IN ('Kempton' , 'Lingfield',
        'Southwell',
        'Wolverhampton')
GROUP BY b.race_id
HAVING runners BETWEEN 4 AND 16
ORDER BY b.race_id;

CREATE INDEX `idx_my_races_race_id` ON `smartform`.`my_races` (race_id);

/*
 * Create my_runners
 */

DROP TABLE IF EXISTS my_runners;

CREATE TABLE my_runners AS SELECT runner_id,
    a.race_id,
    stall_number,
    gender,
    age,
    trainer_id,
    sire_id,
    dam_id,
    finish_position,
    amended_position,
    unfinished,
    starting_price_decimal,
    tack_blinkers,
    tack_visor,
    tack_cheek_piece,
    tack_tongue_strap FROM
    historic_runners a
        INNER JOIN
    my_races b
WHERE
    a.race_id = b.race_id
ORDER BY race_id , stall_number;

CREATE INDEX `idx_my_runners_runner_id_race_id` ON `smartform`.`my_runners` (runner_id, race_id);
CREATE INDEX `idx_my_runners_race_id` ON `smartform`.`my_runners` (race_id);

-- Include meeting_date in my_runners, for convenience in subsequent calculations

ALTER TABLE my_runners ADD COLUMN meeting_date DATE AFTER race_id;

UPDATE my_runners a
        INNER JOIN
    my_races b ON a.race_id = b.race_id
        AND a.race_id = b.race_id 
SET 
    a.meeting_date = b.meeting_date;

/*
 * Delete (small number of) races, and runners in those races, with aberrant (duplicated or NULL) stall numbers
 */
 
DROP TABLE IF EXISTS my_temp;
 
CREATE TEMPORARY TABLE my_temp AS
    SELECT 
        race_id
    FROM
        my_runners
    GROUP BY race_id , stall_number
    HAVING COUNT(*) > 1
UNION 
SELECT DISTINCT
    (race_id) AS race_id
FROM
    my_runners
WHERE
    stall_number IS NULL;

DELETE FROM my_races 
WHERE
    race_id IN (SELECT 
        race_id
    FROM
        my_temp);
 
DELETE FROM my_runners 
WHERE
    race_id IN (SELECT 
        race_id
    FROM
        my_temp);
      
/*
 * Encode finpos and win
 */

ALTER TABLE my_runners ADD COLUMN finpos TINYINT(3) AFTER unfinished, ADD COLUMN win TINYINT(3) AFTER finpos;

UPDATE my_runners 
SET 
    finpos = CASE
        WHEN unfinished IS NOT NULL THEN 0
        WHEN amended_position IS NULL THEN finish_position
        ELSE amended_position
    END,
    win = CASE
        WHEN finpos = 1 THEN 1
        ELSE 0
    END;

ALTER TABLE my_runners DROP COLUMN finish_position, DROP COLUMN amended_position, DROP COLUMN unfinished;

-- Weed out (small number of) aberrant races with finpos > 16

DROP TABLE IF EXISTS my_temp;
 
CREATE TEMPORARY TABLE my_temp AS
    SELECT 
        DISTINCT(race_id)
    FROM
        my_runners
    WHERE finpos > 16
    GROUP BY race_id;

DELETE FROM my_races 
WHERE
    race_id IN (SELECT 
        race_id
    FROM
        my_temp);
        
DELETE FROM my_runners 
WHERE
    race_id IN (SELECT 
        race_id
    FROM
        my_temp);

/*
 * Assign default values for missing ages, and starting prices (not used in probability models)
 */

-- Assign default value (of average age) to missing ages

DROP TABLE IF EXISTS my_temp;

CREATE TEMPORARY TABLE my_temp AS SELECT 
            ROUND(AVG(age), 0) AS avg_age
        FROM
            my_runners;

UPDATE my_runners 
SET 
    age = (SELECT 
            avg_age
        FROM
            my_temp)
WHERE
    age IS NULL;

-- Weed out races involving horses with aberrant ages (not between 2 and 12 years)

DROP TABLE IF EXISTS my_temp;
 
CREATE TEMPORARY TABLE my_temp AS
    SELECT 
        DISTINCT(race_id)
    FROM
        my_runners
    WHERE age not between 2 and 12
    GROUP BY race_id;

DELETE FROM my_races 
WHERE
    race_id IN (SELECT 
        race_id
    FROM
        my_temp);
        
DELETE FROM my_runners 
WHERE
    race_id IN (SELECT 
        race_id
    FROM
        my_temp);

-- Assign default value (of average starting price) to missing starting prices

ALTER TABLE my_runners ADD COLUMN sp FLOAT AFTER starting_price_decimal;

DROP TABLE IF EXISTS my_temp;

CREATE TEMPORARY TABLE my_temp AS SELECT 
            ROUND(AVG(starting_price_decimal), 1) AS avg_sp
        FROM
            my_runners;

UPDATE my_runners 
SET 
    sp = starting_price_decimal
WHERE
    starting_price_decimal IS NOT NULL;

UPDATE my_runners 
SET 
    sp = (SELECT 
            avg_sp
        FROM
            my_temp)
WHERE
    starting_price_decimal IS NULL;

ALTER TABLE my_runners DROP COLUMN starting_price_decimal;
    
/*
 * Calculate adjusted market probabilities (not used in probability models)
 */

ALTER TABLE my_runners ADD COLUMN mkt_prob FLOAT AFTER sp, ADD COLUMN overage FLOAT AFTER mkt_prob, ADD COLUMN adj_mkt_prob FLOAT AFTER overage;

UPDATE my_runners 
SET 
    mkt_prob = 1 / sp;

DROP TABLE IF EXISTS my_temp;

CREATE TEMPORARY TABLE my_temp AS SELECT
         runner_id, race_id, SUM(mkt_prob) OVER(PARTITION BY race_id) AS overage FROM my_runners;

UPDATE my_runners a
        INNER JOIN
    my_temp b ON a.runner_id = b.runner_id
        AND a.race_id = b.race_id 
SET 
    a.overage = b.overage,
    adj_mkt_prob = mkt_prob / b.overage;
    
/*
 * Calculate win percentages of runners (not used in probability models)
 */

DROP TABLE IF EXISTS my_temp;

CREATE TEMPORARY TABLE my_temp AS
SELECT 
    meeting_date, race_id, runner_id, SUM(win) OVER w AS 'cumulative wins', row_number() OVER w AS 'race count'
FROM my_runners
WINDOW w AS (PARTITION BY runner_id ORDER BY meeting_date, race_id)
ORDER BY runner_id ASC, race_id ASC;

ALTER TABLE my_runners ADD COLUMN win_perc FLOAT AFTER win;

UPDATE my_runners
        INNER JOIN
    my_temp ON my_runners.runner_id = my_temp.runner_id
        AND my_runners.race_id = my_temp.race_id 
SET 
    my_runners.win_perc = my_temp.`cumulative wins` / my_temp.`race count`;

/*
 * Calculate strike rates of sires and dams
 */

-- Note, since sire_sr may be used in models, strike rates should as of latest date *prior to* current meeting date
 
DROP TABLE IF EXISTS my_temp;
 
CREATE TEMPORARY TABLE my_temp AS SELECT 
    a.runner_id, a.race_id, b.win_perc AS sire_sr
FROM
    my_runners a
        LEFT JOIN
    my_runners b ON a.sire_id = b.runner_id
        AND b.meeting_date = (SELECT 
            MAX(c.meeting_date)
        FROM
            my_runners c
        WHERE
            a.sire_id = c.runner_id
                AND c.meeting_date < a.meeting_date);

ALTER TABLE my_runners ADD COLUMN sire_sr FLOAT AFTER sire_id;

UPDATE my_runners a
        INNER JOIN
    my_temp b ON a.runner_id = b.runner_id
        AND a.race_id = b.race_id 
SET 
    a.sire_sr = b.sire_sr;

-- Note, since dam_sr may be used in models, strike rates should as of latest date *prior to* current meeting date

DROP TABLE IF EXISTS my_temp;
 
CREATE TEMPORARY TABLE my_temp AS SELECT 
    a.runner_id, a.race_id, b.win_perc AS dam_sr
FROM
    my_runners a
        LEFT JOIN
    my_runners b ON a.dam_id = b.runner_id
        AND b.meeting_date = (SELECT 
            MAX(c.meeting_date)
        FROM
            my_runners c
        WHERE
            a.dam_id = c.runner_id
                AND c.meeting_date < a.meeting_date);

ALTER TABLE my_runners ADD COLUMN dam_sr FLOAT AFTER dam_id;

UPDATE my_runners a
        INNER JOIN
    my_temp b ON a.runner_id = b.runner_id
        AND a.race_id = b.race_id 
SET 
    a.dam_sr = b.dam_sr;
 
 /*
  * Calculate strike rates of trainers
 */

-- Note, since trainer_sr may be used in models, strike rates should as of latest date *prior to* current meeting date

-- Collapse multiple races on same meeting date

DROP TABLE IF EXISTS trainer_step1;

CREATE TEMPORARY TABLE trainer_step1 AS SELECT trainer_id,
    meeting_date,
    SUM(win) AS wins,
    COUNT(*) AS races FROM
    my_runners
GROUP BY trainer_id , meeting_date
ORDER BY trainer_id , meeting_date;

-- Calculate cumulative wins and race count as of meeting date
DROP TABLE IF EXISTS trainer_step2;

CREATE TEMPORARY TABLE trainer_step2 AS
SELECT 
    trainer_id, meeting_date, SUM(wins) OVER w AS 'cumulative wins', sum(races) OVER w AS 'race count'
FROM trainer_step1
WINDOW w AS (PARTITION BY trainer_id ORDER BY meeting_date)
ORDER BY trainer_id, meeting_date;

-- Calculate cumulative wins and race count prior to meeting date

DROP TABLE IF EXISTS trainer_step3;

CREATE TEMPORARY TABLE trainer_step3 AS SELECT
  trainer_id, meeting_date,
  LAG(a.meeting_date) OVER w as 'lagged meeting date',
  LAG(a.`cumulative wins`) OVER w AS 'lagged cumulative wins',
  LAG(a.`race count`) OVER w AS 'lagged race count'
FROM
  trainer_step2 a WINDOW w AS (
    PARTITION BY trainer_id
    ORDER BY
      meeting_date
  )
ORDER BY
  trainer_id,
  meeting_date;

CREATE INDEX `idx_trainer_step3_trainer_id_meeting_date` ON `smartform`.`trainer_step3` (trainer_id, meeting_date);

-- Incoporate trainer strikes rates into my_runners

ALTER TABLE my_runners ADD COLUMN trainer_sr FLOAT AFTER trainer_id;

CREATE INDEX `idx_my_runners_trainer_id_meeting_date` ON `smartform`.`my_runners` (trainer_id, meeting_date);

UPDATE my_runners a
        LEFT JOIN
    trainer_step3 b ON a.trainer_id = b.trainer_id
        AND a.meeting_date = b.meeting_date 
SET 
    a.trainer_sr = b.`lagged cumulative wins` / b.`lagged race count`;

 /*
  * Delete (small number of) races, and runners in those races, with missing trainer_id, sire_id, dam_id, or gender; or with aberrant gender (i.e., not defined in manual)
  */

DROP TABLE IF EXISTS my_temp;

CREATE TEMPORARY TABLE my_temp AS
SELECT DISTINCT(race_id) FROM my_runners WHERE trainer_ID IS NULL OR sire_id IS NULL or dam_id IS NULL or gender IS NULL or gender not in ('G', 'F', 'M', 'C', 'H');

DELETE FROM my_races 
WHERE
    race_id IN (SELECT 
        race_id
    FROM
        my_temp);
 
DELETE FROM my_runners 
WHERE
    race_id IN (SELECT 
        race_id
    FROM
        my_temp);