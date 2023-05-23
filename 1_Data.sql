-- This script takes about 47 minutes to execute

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
        AND COALESCE(all_weather, -1) = 1
        AND COALESCE(race_type, '') = 'All Weather Flat'
        AND COALESCE(course, '') IN ('Kempton', 'Lingfield', 'Southwell', 'Wolverhampton')
        AND COALESCE(maiden, -1) = 0
        AND COALESCE(class, -1) IN (1, 2, 3, 4, 5)
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
    jockey_id,
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

-- Include meeting_date in my_runners, for convenience in subsequent calculations

ALTER TABLE my_runners ADD COLUMN meeting_date DATE AFTER race_id;

UPDATE my_runners a
        INNER JOIN
    my_races b ON a.race_id = b.race_id
        AND a.race_id = b.race_id 
SET 
    a.meeting_date = b.meeting_date;

CREATE INDEX `idx_my_runners_runner_id` ON `smartform`.`my_runners` (runner_id);
CREATE INDEX `idx_my_runners_race_id` ON `smartform`.`my_runners` (race_id);
CREATE INDEX `idx_my_runners_jockey_id` ON `smartform`.`my_runners` (jockey_id);
CREATE INDEX `idx_my_runners_trainer_id` ON `smartform`.`my_runners` (trainer_id);
CREATE INDEX `idx_my_runners_meeting_date` ON `smartform`.`my_runners` (meeting_date);

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
  * Calculate strike rates of trainers (All Weather courses only)
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

--------------------------------------------------------------------------------
-- Create additional variables for model MG1 and other versions of this model --
--------------------------------------------------------------------------------

/*
 * Create temporary table for the calculation of various strikes rates
 */

DROP TEMPORARY TABLE IF EXISTS my_temp;

CREATE TEMPORARY TABLE my_temp AS 
SELECT 
    subquery.race_id, 
    subquery.meeting_date, 
    subquery.runner_id, 
    subquery.jockey_id,
    subquery.trainer_id,
    subquery.KEM,
    subquery.LIN,
    subquery.SOU,
    subquery.WOL,
    subquery.finpos,
    CASE
        WHEN subquery.finpos = 1 THEN 1
        ELSE 0
    END as win,
    subquery.le1mi,
    subquery.class
FROM 
(
    SELECT 
        historic_races.race_id, 
        historic_races.meeting_date, 
        historic_runners.runner_id, 
        historic_runners.jockey_id,
        historic_runners.trainer_id,
        CASE
            WHEN course = 'Kempton' THEN 1
            ELSE 0
        END as KEM,
        CASE
            WHEN course = 'Lingfield' THEN 1
            ELSE 0
        END AS LIN,
        CASE
            WHEN course = 'Southwell' THEN 1
            ELSE 0
        END as SOU,
        CASE
            WHEN course = 'Wolverhampton' THEN 1
            ELSE 0
        END as WOL,
        CASE
            WHEN historic_runners.unfinished IS NOT NULL THEN 0
            WHEN historic_runners.amended_position IS NULL THEN historic_runners.finish_position
            ELSE historic_runners.amended_position
        END as finpos,
        CASE
            WHEN historic_races.distance_yards <= 1760 THEN 1
            ELSE 0
        END as le1mi,
        historic_races.class
    FROM 
        historic_races
    INNER JOIN 
        historic_runners
    ON 
        historic_races.race_id = historic_runners.race_id
    WHERE
        historic_runners.runner_id IN (SELECT runner_id FROM my_runners)
        OR
        historic_runners.jockey_id IN (SELECT jockey_id FROM my_runners)
        OR
        historic_runners.trainer_id IN (SELECT trainer_id FROM my_runners)
) as subquery;

CREATE INDEX idx_meeting_date_id ON my_temp(meeting_date);

CREATE INDEX idx_runner_id_meeting_date ON my_temp(runner_id, meeting_date);
CREATE INDEX idx_runner_id_class ON my_temp(runner_id, class);
CREATE INDEX idx_runner_id_KEM_le1mi ON my_temp(runner_id, KEM, le1mi);
CREATE INDEX idx_runner_id_LIN_le1mi ON my_temp(runner_id, LIN, le1mi);
CREATE INDEX idx_runner_id_SOU_le1mi ON my_temp(runner_id, SOU, le1mi);
CREATE INDEX idx_runner_id_WOL_le1mi ON my_temp(runner_id, WOL, le1mi);

CREATE INDEX idx_jockey_id_meeting_date ON my_temp(jockey_id, meeting_date);
CREATE INDEX idx_jockey_id_KEM ON my_temp(jockey_id, KEM);
CREATE INDEX idx_jockey_id_LIN ON my_temp(jockey_id, LIN);
CREATE INDEX idx_jockey_id_SOU ON my_temp(jockey_id, SOU);
CREATE INDEX idx_jockey_id_WOL ON my_temp(jockey_id, WOL);

CREATE INDEX idx_trainer_id ON my_temp(trainer_id);
CREATE INDEX idx_meeting_date ON my_temp(meeting_date);
CREATE INDEX idx_trainer_id_KEM ON my_temp(trainer_id, KEM);
CREATE INDEX idx_trainer_id_LIN ON my_temp(trainer_id, LIN);
CREATE INDEX idx_trainer_id_SOU ON my_temp(trainer_id, SOU);
CREATE INDEX idx_trainer_id_WOL ON my_temp(trainer_id, WOL);

/*
 * Calculate each runner's 30-day strike rate over all (not just All Weather) races
 */

-- Note, since sr_30 may be used in models, strike rates should as of latest date *prior to* current meeting date

-- Add new column if it does not exist
ALTER TABLE smartform.my_runners
ADD COLUMN sr_30d FLOAT;

-- Calculate and update the strike rate
UPDATE smartform.my_runners r
SET sr_30d = (
    SELECT 
        COALESCE(SUM(t.win) / COUNT(*), 0)
    FROM 
        my_temp t
    WHERE 
        r.runner_id = t.runner_id
        AND t.meeting_date BETWEEN DATE_SUB(r.meeting_date, INTERVAL 1 MONTH) AND DATE_SUB(r.meeting_date, INTERVAL 1 DAY)
);

/*
 * Calculate each runner's lifetime strike rate over all (not just All Weather) races, broken out by class (1-5)
 */

-- Note, since these new columns may be used in models, strike rates should as of latest date *prior to* current meeting date

ALTER TABLE smartform.my_runners
ADD COLUMN sr_lifetime_class_5 FLOAT;

UPDATE smartform.my_runners r
SET sr_lifetime_class_5 = (
    SELECT 
        COALESCE(SUM(t.win) / COUNT(*), 0)
    FROM 
        my_temp t
    WHERE 
        r.runner_id = t.runner_id
        AND t.class = 5
        AND t.meeting_date < r.meeting_date
);

ALTER TABLE smartform.my_runners
ADD COLUMN sr_lifetime_class_4 FLOAT;

UPDATE smartform.my_runners r
SET sr_lifetime_class_4 = (
    SELECT 
        COALESCE(SUM(t.win) / COUNT(*), 0)
    FROM 
        my_temp t
    WHERE 
        r.runner_id = t.runner_id
        AND t.class = 4
        AND t.meeting_date < r.meeting_date
);

ALTER TABLE smartform.my_runners
ADD COLUMN sr_lifetime_class_3 FLOAT;

UPDATE smartform.my_runners r
SET sr_lifetime_class_3 = (
    SELECT 
        COALESCE(SUM(t.win) / COUNT(*), 0)
    FROM 
        my_temp t
    WHERE 
        r.runner_id = t.runner_id
        AND t.class = 3
        AND t.meeting_date < r.meeting_date
);

ALTER TABLE smartform.my_runners
ADD COLUMN sr_lifetime_class_2 FLOAT;

UPDATE smartform.my_runners r
SET sr_lifetime_class_2 = (
    SELECT 
        COALESCE(SUM(t.win) / COUNT(*), 0)
    FROM 
        my_temp t
    WHERE 
        r.runner_id = t.runner_id
        AND t.class = 2
        AND t.meeting_date < r.meeting_date
);

ALTER TABLE smartform.my_runners
ADD COLUMN sr_lifetime_class_1 FLOAT;

UPDATE smartform.my_runners r
SET sr_lifetime_class_1 = (
    SELECT 
        COALESCE(SUM(t.win) / COUNT(*), 0)
    FROM 
        my_temp t
    WHERE 
        r.runner_id = t.runner_id
        AND t.class = 1
        AND t.meeting_date < r.meeting_date
);

/*
 * Calculate each jockey's lifetime strike rate for each All Weather course (Kempton, Lingfield, Southwell, Wolverhampton)
 */

-- Note, since these new columns may be used in models, strike rates should as of latest date *prior to* current meeting date

ALTER TABLE smartform.my_runners
ADD COLUMN jockey_sr_KEM FLOAT;

DROP TEMPORARY TABLE IF EXISTS tmp_jockey_sr_KEM;

CREATE TEMPORARY TABLE tmp_jockey_sr_KEM AS
SELECT 
    jockey_id, 
    COALESCE(SUM(win) / COUNT(*), 0) as win_rate,
    meeting_date
FROM 
    my_temp
WHERE 
    KEM = 1
GROUP BY 
    jockey_id,
    meeting_date;

CREATE INDEX idx_tmp_jockey_id ON tmp_jockey_sr_KEM(jockey_id);

UPDATE 
    smartform.my_runners r
INNER JOIN
    tmp_jockey_sr_KEM t
ON
    r.jockey_id = t.jockey_id AND t.meeting_date < r.meeting_date
SET 
    r.jockey_sr_KEM = t.win_rate;

ALTER TABLE smartform.my_runners
ADD COLUMN jockey_sr_LIN FLOAT;

DROP TEMPORARY TABLE IF EXISTS tmp_jockey_sr_LIN;

CREATE TEMPORARY TABLE tmp_jockey_sr_LIN AS
SELECT 
    jockey_id, 
    COALESCE(SUM(win) / COUNT(*), 0) as win_rate,
    meeting_date
FROM 
    my_temp
WHERE 
    LIN = 1
GROUP BY 
    jockey_id,
    meeting_date;

CREATE INDEX idx_tmp_jockey_id ON tmp_jockey_sr_LIN(jockey_id);

UPDATE 
    smartform.my_runners r
INNER JOIN
    tmp_jockey_sr_LIN t
ON
    r.jockey_id = t.jockey_id AND t.meeting_date < r.meeting_date
SET 
    r.jockey_sr_LIN = t.win_rate;

ALTER TABLE smartform.my_runners
ADD COLUMN jockey_sr_SOU FLOAT;

DROP TEMPORARY TABLE IF EXISTS tmp_jockey_sr_SOU;

CREATE TEMPORARY TABLE tmp_jockey_sr_SOU AS
SELECT 
    jockey_id, 
    COALESCE(SUM(win) / COUNT(*), 0) as win_rate,
    meeting_date
FROM 
    my_temp
WHERE 
    SOU = 1
GROUP BY 
    jockey_id,
    meeting_date;

CREATE INDEX idx_tmp_jockey_id ON tmp_jockey_sr_SOU(jockey_id);

UPDATE 
    smartform.my_runners r
INNER JOIN
    tmp_jockey_sr_SOU t
ON
    r.jockey_id = t.jockey_id AND t.meeting_date < r.meeting_date
SET 
    r.jockey_sr_SOU = t.win_rate;

ALTER TABLE smartform.my_runners
ADD COLUMN jockey_sr_WOL FLOAT;

DROP TEMPORARY TABLE IF EXISTS tmp_jockey_sr_WOL;

CREATE TEMPORARY TABLE tmp_jockey_sr_WOL AS
SELECT 
    jockey_id, 
    COALESCE(SUM(win) / COUNT(*), 0) as win_rate,
    meeting_date
FROM 
    my_temp
WHERE 
    WOL = 1
GROUP BY 
    jockey_id,
    meeting_date;

CREATE INDEX idx_tmp_jockey_id ON tmp_jockey_sr_WOL(jockey_id);

UPDATE 
    smartform.my_runners r
INNER JOIN
    tmp_jockey_sr_WOL t
ON
    r.jockey_id = t.jockey_id AND t.meeting_date < r.meeting_date
SET 
    r.jockey_sr_WOL = t.win_rate;

/*
 * Calculate each trainer's lifetime strike rate for each All Weather course (Kempton, Lingfield, Southwell, Wolverhampton)
 */

-- Note, since these new columns may be used in models, strike rates should as of latest date *prior to* current meeting date

ALTER TABLE smartform.my_runners
ADD COLUMN trainer_sr_KEM FLOAT;

DROP TEMPORARY TABLE IF EXISTS tmp_trainer_sr_KEM;

CREATE TEMPORARY TABLE tmp_trainer_sr_KEM AS
SELECT 
    trainer_id, 
    COALESCE(SUM(win) / COUNT(*), 0) as win_rate,
    meeting_date
FROM 
    my_temp
WHERE 
    KEM = 1
GROUP BY 
    trainer_id,
    meeting_date;

CREATE INDEX idx_tmp_trainer_id ON tmp_trainer_sr_KEM(trainer_id);

UPDATE 
    smartform.my_runners r
INNER JOIN
    tmp_trainer_sr_KEM t
ON
    r.trainer_id = t.trainer_id AND t.meeting_date < r.meeting_date
SET 
    r.trainer_sr_KEM = t.win_rate;

ALTER TABLE smartform.my_runners
ADD COLUMN trainer_sr_LIN FLOAT;

DROP TEMPORARY TABLE IF EXISTS tmp_trainer_sr_LIN;

CREATE TEMPORARY TABLE tmp_trainer_sr_LIN AS
SELECT 
    trainer_id, 
    COALESCE(SUM(win) / COUNT(*), 0) as win_rate,
    meeting_date
FROM 
    my_temp
WHERE 
    LIN = 1
GROUP BY 
    trainer_id,
    meeting_date;

CREATE INDEX idx_tmp_trainer_id ON tmp_trainer_sr_LIN(trainer_id);

UPDATE 
    smartform.my_runners r
INNER JOIN
    tmp_trainer_sr_LIN t
ON
    r.trainer_id = t.trainer_id AND t.meeting_date < r.meeting_date
SET 
    r.trainer_sr_LIN = t.win_rate;

ALTER TABLE smartform.my_runners
ADD COLUMN trainer_sr_SOU FLOAT;

DROP TEMPORARY TABLE IF EXISTS tmp_trainer_sr_SOU;

CREATE TEMPORARY TABLE tmp_trainer_sr_SOU AS
SELECT 
    trainer_id, 
    COALESCE(SUM(win) / COUNT(*), 0) as win_rate,
    meeting_date
FROM 
    my_temp
WHERE 
    SOU = 1
GROUP BY 
    trainer_id,
    meeting_date;

CREATE INDEX idx_tmp_trainer_id ON tmp_trainer_sr_SOU(trainer_id);

UPDATE 
    smartform.my_runners r
INNER JOIN
    tmp_trainer_sr_SOU t
ON
    r.trainer_id = t.trainer_id AND t.meeting_date < r.meeting_date
SET 
    r.trainer_sr_SOU = t.win_rate;

ALTER TABLE smartform.my_runners
ADD COLUMN trainer_sr_WOL FLOAT;

DROP TEMPORARY TABLE IF EXISTS tmp_trainer_sr_WOL;

CREATE TEMPORARY TABLE tmp_trainer_sr_WOL AS
SELECT 
    trainer_id, 
    COALESCE(SUM(win) / COUNT(*), 0) as win_rate,
    meeting_date
FROM 
    my_temp
WHERE 
    WOL = 1
GROUP BY 
    trainer_id,
    meeting_date;

CREATE INDEX idx_tmp_trainer_id ON tmp_trainer_sr_WOL(trainer_id);

UPDATE 
    smartform.my_runners r
INNER JOIN
    tmp_trainer_sr_WOL t
ON
    r.trainer_id = t.trainer_id AND t.meeting_date < r.meeting_date
SET 
    r.trainer_sr_WOL = t.win_rate;
    
/*
 * Calculate each jockey's 30-day strike rate over all (not just All Weather) races
 */

-- Note, since these new columns may be used in models, strike rates should as of latest date *prior to* current meeting date

ALTER TABLE smartform.my_runners
ADD COLUMN jockey_sr_30d FLOAT;

DROP PROCEDURE IF EXISTS update_jockey_sr_30d;

DELIMITER $$

CREATE PROCEDURE update_jockey_sr_30d()
BEGIN
    DECLARE done BOOLEAN DEFAULT FALSE;
    DECLARE v_jockey_id INT;
    DECLARE v_meeting_date DATE;
    DECLARE v_strike_rate FLOAT;
    DECLARE cur CURSOR FOR SELECT jockey_id, meeting_date FROM my_runners;
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;

    OPEN cur;

    read_loop: LOOP
        FETCH cur INTO v_jockey_id, v_meeting_date;
        IF done THEN
            LEAVE read_loop;
        END IF;
        
        -- calculate the 30-day strike rate
        SELECT COALESCE(SUM(t.win) / COUNT(*), 0) INTO v_strike_rate
        FROM my_temp t
        WHERE t.jockey_id = v_jockey_id
        AND t.meeting_date BETWEEN v_meeting_date - INTERVAL 30 DAY AND v_meeting_date - INTERVAL 1 DAY;

        -- update the my_runners table
        UPDATE my_runners
        SET jockey_sr_30d = v_strike_rate
        WHERE jockey_id = v_jockey_id AND meeting_date = v_meeting_date;

    END LOOP;

    CLOSE cur;
END;
$$

DELIMITER ;

CALL update_jockey_sr_30d();

/*
 * Calculate each trainer's 30-day strike rate over all (not just All Weather) races
 */

-- Note, since these new columns may be used in models, strike rates should as of latest date *prior to* current meeting date

ALTER TABLE smartform.my_runners
ADD COLUMN trainer_sr_30d FLOAT;

DROP PROCEDURE IF EXISTS update_trainer_sr_30d;

DELIMITER $$

CREATE PROCEDURE update_trainer_sr_30d()
BEGIN
    DECLARE done BOOLEAN DEFAULT FALSE;
    DECLARE v_trainer_id INT;
    DECLARE v_meeting_date DATE;
    DECLARE v_strike_rate FLOAT;
    DECLARE batch_size INT DEFAULT 1000;
    DECLARE batch_counter INT DEFAULT 0;
    
    DECLARE cur CURSOR FOR SELECT trainer_id, meeting_date FROM my_runners;
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;
    
    OPEN cur;

    read_loop: LOOP
        FETCH cur INTO v_trainer_id, v_meeting_date;
        IF done THEN
            LEAVE read_loop;
        END IF;
        
        -- calculate the 30-day strike rate
        SELECT COALESCE(SUM(t.win) / COUNT(*), 0) INTO v_strike_rate
        FROM my_temp t
        WHERE t.trainer_id = v_trainer_id
        AND t.meeting_date BETWEEN v_meeting_date - INTERVAL 30 DAY AND v_meeting_date - INTERVAL 1 DAY;

        -- update the my_runners table
        UPDATE my_runners
        SET trainer_sr_30d = v_strike_rate
        WHERE trainer_id = v_trainer_id AND meeting_date = v_meeting_date;
        
        SET batch_counter = batch_counter + 1;
        
        IF batch_counter >= batch_size THEN
            COMMIT;
            SET batch_counter = 0;
        END IF;
    END LOOP;

    CLOSE cur;
END;
$$

DELIMITER ;

CALL update_trainer_sr_30d();

/*
 * Calculate runner's "form" features for races less than or equal to one mile
 */

-- Note, since these new columns may be used in models, "form" features should as of latest date *prior to* current meeting date

-- Kempton

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior1_KEM_le1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior1_KEM_le1mi = (
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
        my_temp t
    WHERE 
        r.runner_id = t.runner_id
        AND t.KEM = 1
        AND t.le1mi = 1
        AND t.meeting_date < r.meeting_date
    ORDER BY 
        t.meeting_date DESC
    LIMIT 1
);

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior2_KEM_le1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior2_KEM_le1mi = COALESCE((
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
    (
        SELECT 
            runner_id, 
            CASE 
                WHEN finpos <= 4 THEN finpos
                ELSE 0
            END as finpos,
            ROW_NUMBER() OVER (PARTITION BY runner_id ORDER BY meeting_date DESC) as rn
        FROM 
            my_temp t
        WHERE 
            KEM = 1
            AND t.le1mi = 1
    ) t
    WHERE 
        r.runner_id = t.runner_id
        AND t.rn = 2
), 0);

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior3_KEM_le1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior3_KEM_le1mi = COALESCE((
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
    (
        SELECT 
            runner_id, 
            CASE 
                WHEN finpos <= 4 THEN finpos
                ELSE 0
            END as finpos,
            ROW_NUMBER() OVER (PARTITION BY runner_id ORDER BY meeting_date DESC) as rn
        FROM 
            my_temp t
        WHERE 
            KEM = 1
            AND t.le1mi = 1
    ) t
    WHERE 
        r.runner_id = t.runner_id
        AND t.rn = 3
), 0);

-- Lingfield

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior1_LIN_le1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior1_LIN_le1mi = (
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
        my_temp t
    WHERE 
        r.runner_id = t.runner_id
        AND t.LIN = 1
        AND t.le1mi = 1
        AND t.meeting_date < r.meeting_date
    ORDER BY 
        t.meeting_date DESC
    LIMIT 1
);

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior2_LIN_le1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior2_LIN_le1mi = COALESCE((
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
    (
        SELECT 
            runner_id, 
            CASE 
                WHEN finpos <= 4 THEN finpos
                ELSE 0
            END as finpos,
            ROW_NUMBER() OVER (PARTITION BY runner_id ORDER BY meeting_date DESC) as rn
        FROM 
            my_temp t
        WHERE 
            LIN = 1
            AND t.le1mi = 1
    ) t
    WHERE 
        r.runner_id = t.runner_id
        AND t.rn = 2
), 0);

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior3_LIN_le1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior3_LIN_le1mi = COALESCE((
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
    (
        SELECT 
            runner_id, 
            CASE 
                WHEN finpos <= 4 THEN finpos
                ELSE 0
            END as finpos,
            ROW_NUMBER() OVER (PARTITION BY runner_id ORDER BY meeting_date DESC) as rn
        FROM 
            my_temp
        WHERE 
            LIN = 1
            AND t.le1mi = 1
    ) t
    WHERE 
        r.runner_id = t.runner_id
        AND t.rn = 3
), 0);

-- Southwell

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior1_SOU_le1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior1_SOU_le1mi = (
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
        my_temp t
    WHERE 
        r.runner_id = t.runner_id
        AND t.SOU = 1
        AND t.le1mi = 1
        AND t.meeting_date < r.meeting_date
    ORDER BY 
        t.meeting_date DESC
    LIMIT 1
);

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior2_SOU_le1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior2_SOU_le1mi = COALESCE((
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
    (
        SELECT 
            runner_id, 
            CASE 
                WHEN finpos <= 4 THEN finpos
                ELSE 0
            END as finpos,
            ROW_NUMBER() OVER (PARTITION BY runner_id ORDER BY meeting_date DESC) as rn
        FROM 
            my_temp t
        WHERE 
            SOU = 1
            AND t.le1mi = 1
    ) t
    WHERE 
        r.runner_id = t.runner_id
        AND t.rn = 2
), 0);

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior3_SOU_le1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior3_SOU_le1mi = COALESCE((
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
    (
        SELECT 
            runner_id, 
            CASE 
                WHEN finpos <= 4 THEN finpos
                ELSE 0
            END as finpos,
            ROW_NUMBER() OVER (PARTITION BY runner_id ORDER BY meeting_date DESC) as rn
        FROM 
            my_temp t
        WHERE 
            SOU = 1
            AND t.le1mi = 1
    ) t
    WHERE 
        r.runner_id = t.runner_id
        AND t.rn = 3
), 0);

-- Wolverhampton

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior1_WOL_le1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior1_WOL_le1mi = (
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
        my_temp t
    WHERE 
        r.runner_id = t.runner_id
        AND t.WOL = 1
        AND t.le1mi = 1
        AND t.meeting_date < r.meeting_date
    ORDER BY 
        t.meeting_date DESC
    LIMIT 1
);

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior2_WOL_le1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior2_WOL_le1mi = COALESCE((
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
    (
        SELECT 
            runner_id, 
            CASE 
                WHEN finpos <= 4 THEN finpos
                ELSE 0
            END as finpos,
            ROW_NUMBER() OVER (PARTITION BY runner_id ORDER BY meeting_date DESC) as rn
        FROM 
            my_temp t
        WHERE 
            WOL = 1
            AND t.le1mi = 1
    ) t
    WHERE 
        r.runner_id = t.runner_id
        AND t.rn = 2
), 0);

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior3_WOL_le1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior3_WOL_le1mi = COALESCE((
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
    (
        SELECT 
            runner_id, 
            CASE 
                WHEN finpos <= 4 THEN finpos
                ELSE 0
            END as finpos,
            ROW_NUMBER() OVER (PARTITION BY runner_id ORDER BY meeting_date DESC) as rn
        FROM 
            my_temp t
        WHERE 
            WOL = 1
            AND t.le1mi = 1
    ) t
    WHERE 
        r.runner_id = t.runner_id
        AND t.rn = 3
), 0);

/*
 * Calculate runner's "form" features for races greater than one mile
 */

-- Note, since these new columns may be used in models, "form" features should as of latest date *prior to* current meeting date

-- Kempton

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior1_KEM_gt1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior1_KEM_gt1mi = (
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
        my_temp t
    WHERE 
        r.runner_id = t.runner_id
        AND t.KEM = 1
        AND t.le1mi = 0
        AND t.meeting_date < r.meeting_date
    ORDER BY 
        t.meeting_date DESC
    LIMIT 1
);

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior2_KEM_gt1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior2_KEM_gt1mi = COALESCE((
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
    (
        SELECT 
            runner_id, 
            CASE 
                WHEN finpos <= 4 THEN finpos
                ELSE 0
            END as finpos,
            ROW_NUMBER() OVER (PARTITION BY runner_id ORDER BY meeting_date DESC) as rn
        FROM 
            my_temp t
        WHERE 
            KEM = 1
            AND t.le1mi = 0
    ) t
    WHERE 
        r.runner_id = t.runner_id
        AND t.rn = 2
), 0);

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior3_KEM_gt1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior3_KEM_gt1mi = COALESCE((
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
    (
        SELECT 
            runner_id, 
            CASE 
                WHEN finpos <= 4 THEN finpos
                ELSE 0
            END as finpos,
            ROW_NUMBER() OVER (PARTITION BY runner_id ORDER BY meeting_date DESC) as rn
        FROM 
            my_temp t
        WHERE 
            KEM = 1
            AND t.le1mi = 0
    ) t
    WHERE 
        r.runner_id = t.runner_id
        AND t.rn = 3
), 0);

-- Lingfield

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior1_LIN_gt1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior1_LIN_gt1mi = (
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
        my_temp t
    WHERE 
        r.runner_id = t.runner_id
        AND t.LIN = 1
        AND t.le1mi = 0
        AND t.meeting_date < r.meeting_date
    ORDER BY 
        t.meeting_date DESC
    LIMIT 1
);

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior2_LIN_gt1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior2_LIN_gt1mi = COALESCE((
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
    (
        SELECT 
            runner_id, 
            CASE 
                WHEN finpos <= 4 THEN finpos
                ELSE 0
            END as finpos,
            ROW_NUMBER() OVER (PARTITION BY runner_id ORDER BY meeting_date DESC) as rn
        FROM 
            my_temp t
        WHERE 
            LIN = 1
            AND t.le1mi = 0
    ) t
    WHERE 
        r.runner_id = t.runner_id
        AND t.rn = 2
), 0);

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior3_LIN_gt1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior3_LIN_gt1mi = COALESCE((
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
    (
        SELECT 
            runner_id, 
            CASE 
                WHEN finpos <= 4 THEN finpos
                ELSE 0
            END as finpos,
            ROW_NUMBER() OVER (PARTITION BY runner_id ORDER BY meeting_date DESC) as rn
        FROM 
            my_temp t
        WHERE 
            LIN = 1
            AND t.le1mi = 0
    ) t
    WHERE 
        r.runner_id = t.runner_id
        AND t.rn = 3
), 0);

-- Southwell

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior1_SOU_gt1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior1_SOU_gt1mi = (
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
        my_temp t
    WHERE 
        r.runner_id = t.runner_id
        AND t.SOU = 1
        AND t.le1mi = 0
        AND t.meeting_date < r.meeting_date
    ORDER BY 
        t.meeting_date DESC
    LIMIT 1
);

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior2_SOU_gt1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior2_SOU_gt1mi = COALESCE((
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
    (
        SELECT 
            runner_id, 
            CASE 
                WHEN finpos <= 4 THEN finpos
                ELSE 0
            END as finpos,
            ROW_NUMBER() OVER (PARTITION BY runner_id ORDER BY meeting_date DESC) as rn
        FROM 
            my_temp t
        WHERE 
            SOU = 1
            AND t.le1mi = 0
    ) t
    WHERE 
        r.runner_id = t.runner_id
        AND t.rn = 2
), 0);

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior3_SOU_gt1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior3_SOU_gt1mi = COALESCE((
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
    (
        SELECT 
            runner_id, 
            CASE 
                WHEN finpos <= 4 THEN finpos
                ELSE 0
            END as finpos,
            ROW_NUMBER() OVER (PARTITION BY runner_id ORDER BY meeting_date DESC) as rn
        FROM 
            my_temp t
        WHERE
            SOU = 1
            AND t.le1mi = 0
    ) t
    WHERE 
        r.runner_id = t.runner_id
        AND t.rn = 3
), 0);

-- Wolverhampton

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior1_WOL_gt1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior1_WOL_gt1mi = (
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
        my_temp t
    WHERE 
        r.runner_id = t.runner_id
        AND t.WOL = 1
        AND t.le1mi = 0
        AND t.meeting_date < r.meeting_date
    ORDER BY 
        t.meeting_date DESC
    LIMIT 1
);

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior2_WOL_gt1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior2_WOL_gt1mi = COALESCE((
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
    (
        SELECT 
            runner_id, 
            CASE 
                WHEN finpos <= 4 THEN finpos
                ELSE 0
            END as finpos,
            ROW_NUMBER() OVER (PARTITION BY runner_id ORDER BY meeting_date DESC) as rn
        FROM 
            my_temp t
        WHERE 
            WOL = 1
            AND t.le1mi = 0
    ) t
    WHERE 
        r.runner_id = t.runner_id
        AND t.rn = 2
), 0);

ALTER TABLE smartform.my_runners
ADD COLUMN pos_prior3_WOL_gt1mi TINYINT;

UPDATE smartform.my_runners r
SET pos_prior3_WOL_gt1mi = COALESCE((
    SELECT 
        CASE 
            WHEN t.finpos <= 4 THEN t.finpos
            ELSE 0
        END
    FROM 
    (
        SELECT 
            runner_id, 
            CASE 
                WHEN finpos <= 4 THEN finpos
                ELSE 0
            END as finpos,
            ROW_NUMBER() OVER (PARTITION BY runner_id ORDER BY meeting_date DESC) as rn
        FROM 
            my_temp t
        WHERE 
            WOL = 1
            AND t.le1mi = 0
    ) t
    WHERE 
        r.runner_id = t.runner_id
        AND t.rn = 3
), 0);