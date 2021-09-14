-------------------------------------------
-- find basic statistics of days differences of reviews for each user
-------------------------------------------
with ratings_time as (
	select *, 
		coalesce(
			unixreviewtime - lag(unixreviewtime) over (partition by gplususerid order by unixreviewtime), 
			interval '0 days'
		) time_diff
	from google_local.public.ratings
	order by gplususerid, unixreviewtime
)
select 
	avg(time_diff), 
	stddev(EXTRACT(epoch FROM time_diff)), -- sd in secs
	min(time_diff), 
	max(time_diff)
from ratings_time

-------------------------------------------
-- data for a histogram of days differences between reviews
-------------------------------------------
select days_diff, count(*)
from (
	select 
	round(extract(epoch from 
		coalesce(
			unixreviewtime - lag(unixreviewtime) over (partition by gplususerid order by unixreviewtime), 
			interval '0 days'
		) / (60*60*24) ) --giorni
	) days_diff
	from google_local.public.ratings
) t
group by 1
order by 1;
-------------------------------------------
-- count reviews per location
-------------------------------------------
select gplusplaceid, count(*)
from public.ratings r
group by 1
having count(*) > 5
order by 2 desc;
-------------------------------------------
-- count reviews per user
-------------------------------------------
select gplususerid, count(*)
from public.ratings r
group by 1
having count(*) > 5
order by 2 desc;
-------------------------------------------
-- filter users and locations with at least 30 reviews
-------------------------------------------
create view filterd_ratings as 
with filter_loc as (
	select gplusplaceid, count(*)
	from public.ratings r
	group by 1
	having count(*) >= 30
), filter_usr as (
	select gplususerid, count(*)
	from public.ratings r
	group by 1
	having count(*) >= 30
)
select row_number() over (), r.* -- row_number creates a review ID on the fly
from public.ratings r
join filter_loc fl using (gplusplaceid)
join filter_usr fu using (gplususerid);
-- Successfully run. Total query runtime: 31 secs 786 msec. 283017 rows affected. --
-------------------------------------------
-- starting from the filtered sample, build co-occurences of reviews within 7 days for each user
-------------------------------------------
select r1.gplususerid "user", r1.gplusplaceid "source",  r2.gplusplaceid "target", r2.unixreviewtime - r1.unixreviewtime as "days diff"
from filterd_ratings r1
join filterd_ratings r2 using (gplususerid)
where r2.unixreviewtime - r1.unixreviewtime < interval '7 days' 
  and r2.unixreviewtime - r1.unixreviewtime > interval '-7 days'
  and r1.row_number <> r2.row_number -- ignore diagonal of cartesian product
order by r1.gplususerid, r1.gplusplaceid, r2.gplusplaceid
-- Successfully run. Total query runtime: 1 min 20 secs. 3794058 rows affected.
-------------------------------------------
-- select the ids of the locations located inside the geographic box Padua - Pordenone
-------------------------------------------
select "gPlusPlaceId", 
	name, 
	address, 
	lat,
	lon
	--ST_SetSRID(ST_MakePoint(lat, lon),4326)
from public.places
where ST_Intersects(
	ST_SetSRID(ST_MakePoint(lat, lon),4326),
	-- box Padova - Pordenone
	ST_MakeEnvelope(11.8, 45.4, 12.7, 46, 4326)
)
