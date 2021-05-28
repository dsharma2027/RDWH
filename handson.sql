create database PraxisJuly;

use [Praxis_july];  --Use is like chdir in py, changing directory; 

--creating table

create table emp
(
id integer,
name varchar(100),    --name is also a keyword, but sql is smart enough to identify it
age smallint,
salary bigint
);

--Viewing data

select * from emp;   --* selects all the columns

select id, name, age 
from emp;  

--Inserting Records(data)
--Records are inserted row by row
insert into emp      --inserting single row
(id,name,age,salary)
values
(1,'Goutham',25,1000); --str has to be enclosed in '' single quotes not in "" double quotes

insert into emp
(id,age,name,salary)  --Column order and value order should match
values
(2,NULL,'Kumar',30000); 

insert into emp
(id,name,salary)  --Column order and value order should match
values
(2,'Nav',30000); --Null doesn't need to be specified explicitly

select * from emp;

--Instead of using Describe table...use Alt+F1(ctrl +F1 in my pc) to see basic st. of Table

--Inserting more rows

insert into emp      
(id,name,age,salary)
values
(4,'A',25,1000),
(5,'B',25,1000),
(6,'C',NULL,1000),
(7, 'D',NULL,5000);

select *from emp;

delete from emp
where 1 = 1; --deletes all records in table

/*insert into emp      
(id,name,age,salary)
values
(4,'A',,1000);  --cant leave without null,if u have specified 4 columns, u shld enter 4 values
*/

--27th Aug session 3

truncate table emp;   --deletes all entries, but table structure isn't altered

insert into emp
values                        --dont give () whole block, bcoz here were are inserting row by row
(1,'Goutham',25,1000),
(2,'Nav',25,2000),
(3,'Mass',25,5000),
(4,'Vicky',NULL,7000),
(5,'Faaru',27,500);

select * from emp;

--filters
select *
from emp
where name ='Goutham';

/* comparison Operators
=
>=
<=
>
<
<> and !=  --not equal,both works 
*/
select *
from emp
where id != 3

select * 
from emp
where name not in ('Goutham', 'Nav');  --Displays name that does'nt have goutham and nav

select * 
from emp
where name in ('Mass', 'Faaru'); --Displays only these name

/*Logical operators
AND
OR
NOT
*/
----------------------------------------------
/* 01 - sep 2020*/

--ddl alter table

alter table emp
alter column [name] varchar(10); -- reducing the size of name column

truncate table emp;

insert into emp
(id, name, age, salary)
values(2,2020, 20, 5000);  ---if it is convertible , it can convert varchar to int

alter table emp
alter column [name] int;

select *
from emp;

truncate table emp;

exec sp_help emp;  --similar to describe table or ctrl +f1

alter table emp
drop column salary;  --dropped salary col.

alter table emp 
add address varchar(50);   --for adding column, no need to speccify column (constraint)

exec sp_help emp;

alter table emp
add city varchar(20),
state  varchar(50),
country varchar(20);  ---multiple column addition

/*Assignment -
create, alter, drop all commands discussed abv
exercise create table, add values, and truncate it , add again, drop table*/

--few classes on order by etc., 
/* 08- sep 2020*/  --from here action are performed on Praxis_july

use Praxis_july;

select * from stud;

select * from stud
order by Age_emp;

select top 5 from stud; --syntax error

select top 5 Id from stud;

select top (3)* from stud; 
--top 3 records it can give any 3 records(random)

exec sp_rename "stud.Age", "Age_emp";  --renaming the column 

exec sp_help stud;

drop table emp;

--limit --limits only certain record for retrival of data
-- select * from stud limit 3; --here it wont work

--copy of table

select * into stud_copy from stud;

truncate table stud_copy;
drop table stud_copy;
GO

--Like,in,between

select * from stud
where age_emp between 23 and 25;  ---23 and 25 inclusive 

select * from stud
where age in (20,23,25);

select * from stud
where name like 'G%'; --displays names that start with G

--_ Pre school'_ _ _ _'
--% Grad school

select * from stud
where name like 'Jim';

select * from stud  
where name between 'A' and 'G';

select * from stud
where age between 25 and 20;  --does'nt work this way

select * from stud
where name like '%a%';
--google about careless

--calculated table --important [dbo].[FIFA19_official_data]use in DS
create table calculate2
(low int,
high int,
myavg as (low+high)/2   --we can have a column with formula
);

insert into calculate2
values 
( 10 , 20 );

select * from calculate;

select * into test2 from calculate;

insert into test
values
(10,20,50);

select * from test

select count(distinct(name))
from stud;
-------------------------------------------------------------------------
/* 15 sep 2020 */

--Import and Export

--Right click on db, tasks, export data(similar to import)

--Agg fn are applied to columns
/*sum, 
average, 
count, 
max , 
min  */

select * 
from stud;

exec sp_rename 'stud.Age_emp', 'Age';

select max(Age) as Max_age
from stud


select avg(Age) as Avg_age
from stud;

select max(Id) as max_id, min(Age) as Min_Age
from stud;


select len(Name) as lent,Name, Age
from stud;

select len(Name) as lent, Name , Age
from stud
where len(Name) >4;

--double quotes -used for aliasing which has space inbetween
select Name "Name of Emp" 
from stud;
----------------------------------------------------------
/* 22 sep 2019*/
--String functions
/*Upper
Lower
len
Ltrim - remove space from left side of string
Rtrim - remove space from right side of string  */

select upper('Goutham Kumar') as name

select lower('Goutham Kumar') as name

select len('Goutham Kumar') as name

select len('  Goutham Kumar  ') as name --2spaces on both sides --len doesn't count the spacces in the right side of the data

select datalength('  Goutham Kumar  ') as name --this calculates the crct spaces on both sides

select len(ltrim('   Goutham Kumar    ')) as name --trims the 2spaces from the left side and reduces the lenght

select len(rtrim('  Goutham Kumar   ')) as name --removes the right spaces

Insert into stud
(Name, Course,Location, Age_emp,Gender)
values
('  Goutham  ', 'ECE','Chennai',0,'M'),
('  Goutahm    ', 'MBA','Pallakad',1,'M');

exec sp_columns stud; 

--variable declaration

declare @a varchar(20) 
set @a = '   Goutham   '
select @a


select @b = 'Gouthammm'---this doesnt work
select @b;

declare @c varchar(20)
set @c = '     Xyz'
select @c
select ltrim(@c)

select getdate();

declare @d datetime
select @d = getdate()
select @d
select upper(@d)--gives month name then date thn yr

declare @e datetime
select @e = getdate()
select @e
select len(@e)

--Advanced lvl string fn
/* substring 
charindex -return charcter index
concat
reverse
replace  */

select substring(Name,1,4) as shortname 
from stud;  --subsets the strng from 1st charcter and prints 4 charcters from that
--here indexing starts with 1

--charindex(exptofind, exptosearch(optional set value)

select CHARINDEX('G', 'The king of Gotham city') as indexvalue;--index of the first occurence

select charindex('G', 'The king of Gotham city is Goutham',9) as indexvalue --now search begins from 9th charcter and first occurence after 9th charcter will be displayed

select substring('Im a monkey',(charindex('monkey','Im a monkey')),6)

select substring('Im a monkey',1,10);

----------------------------------------------------------------------------------------------
/* 29 -sep-2020 */

select * from dbo.stud;

declare @age_inc int --declaration
set @age_inc = 5  --this is new data
select @age_inc as Updated_age,* from stud;

declare @b varchar(100)
set @b = 'C:\Users\Goutham-ROG\Documents\1-Codes\Python - codes\goutham.sql'
select replace(@b,'goutham','flexy') as "repalce value", @b as original_string;

/* assignment
get the file name form the entire file path
C:\Users\Goutham-ROG\Documents\DS studies\College\heelo.py

we shld get hello.py as output
*/

select * from stud;
--getting null values
select isnull(Gender, 'M'),name from stud; --now null is updated


----------------------
/*Assignmnet*/
--1st method -only fns class taught in class 
Declare @path varchar(MAX), @find varchar(MAX), @len_file int, @len_ext int
set @path = 'C:\Users\Goutham-ROG\Documents\DS studies\College\1st sem\RDBMS and Data Warehousing(RDWH- 2.0)\T_SQL Joins.pdf     ' --user i/p here
set @find = @path --copy of filepath
-set @len_file = ltrim(charindex('\',reverse(@find))) --this gives the first position of '\'--to be specific the length of the filename
--set @len_file = (charindex('\\',reverse(@find)) --use this if user specifies filepath with \\ and hash out above line
set @len_ext = ltrim(charindex('.',reverse(@find)))--this gives the length of extension
select reverse(substring(reverse(@find),1,@len_file-1)) as "File name", reverse(substring(reverse(@find),1,@len_ext-1)) as Extension

--2nd method - using right
Declare @path2 varchar(MAX), @find2 varchar(MAX), @len_file2 int, @len_ext2 int
set @path2 = 'C:\Users\Goutham-ROG\Documents\DS studies\College\1st sem\RDBMS and Data Warehousing(RDWH- 2.0)\T_SQL Joins.pdf' --user i/p
set @find2 = @path2 --copy of filepath
set @len_file2 = charindex('\',reverse(@find2))
set @len_ext2 = ltrim(charindex('.',reverse(@find2)))
select right(@find2,@len_file2-1) as "File name", right(@find2,@len_ext2-1) as Extension

--3rd method- Filename without ext
Declare @path3 varchar(MAX), @find3 varchar(MAX), @len_file3 int, @len_ext3 int, @pos int
set @path3 = 'C:\Users\Goutham-ROG\Documents\DS studies\College\1st sem\RDBMS and Data Warehousing(RDWH- 2.0)\T_SQL Joins.pdf' --user i/p
set @find3 = @path3 --copy of filepath
set @len_file3 = ltrim(charindex('\',reverse(@find3)))
set @len_ext3 = ltrim(charindex('.',reverse(@find3)))
--set @pos = (@find3,@len_file3-1)-@len_ext3)
select left(right(@find3,@len_file3-1),(@len_file3-@len_ext3)-1) as "File name without Ext", right(@find3,@len_ext3-1) as Extension

----------------------------------------------
/* 8th oct- 2020*/ --groupby

select * from stud;

select max(age), avg(age) from stud;

select max(age), avg(age), Name from stud;--now the prblm will arise since, there is no agg for name or group by
-- solution--> use grp by or another agg. for name 

--group by
select max(age), avg(age), Name from stud group by name;

--another agg
select max(age), avg(age), max(Name) from stud;

select Location,count(location) as "Count of ppl" 
from stud 
group by location; 

select distinct(Location) from stud; --distinct -> used to remove duplicates

select location from stud group by location; --this groups but doesn't remove the value, it aggregates

--generic question
select top(1) avg(salary), age 
from emp
group by age 
order by salary desc;

--give location for oldest student frm and his name shld have letter s 

select location, name, age
from stud
where age = (select max(age) from stud) --and name like '%s';

--or
--this will mostly don't work coz it is checking max salary and name in same condition adn returns
select location, name, age
from stud
where age = (select max(age) from stud--where name like '%s');

--generic solution
--give dept id, where oldest employee works and his name shld have letter s

select dept_id from emp
where age = (select max(age) from emp) and name like '%s'

----------------------------------------------------------------------------------------------------------
/*13 oct 2020*/
--Joins

create table employee
( id int identity(1,1),
  name varchar(50),
  age smallint,
  salary bigint,
  deptid int
  );
 GO
 
create table deptid
( id int ,
  name varchar(50)
  );
  Go

insert into employee
(name, age, salary,deptid)
values
('Ann', 21,1000,1),
('Muhamad',22,1000,2),
('Lavanaya',23,2000,1),
('Basith',21,2000,3),
('Sruthi',22,3000,4),
('Anchal',21,2000,3);

insert into deptid
values
(1,'HR'),(2,'IT'),(4,'Admin'),(5,'Finance');

select * from employee;
select * from deptid;

--inner join
select *
from employee
inner join deptid
on employee.deptid = deptid.id;

select employee.id, employee.name, employee.deptid, deptid.name
from employee
inner join deptid
on employee.deptid = deptid.id;

--in ssms inner join and join are same here

select employee.id, employee.name, employee.deptid, deptid.name
from employee
join deptid
on employee.deptid = deptid.id;
--(dont use this-- its deprecated)in later versions we need right full keyword

--left join --all entries from left tables and matching values form right table, unmatched values will be null
select employee.id, employee.name, employee.deptid, deptid.name
from employee
left join deptid
on employee.deptid = deptid.id;
--both are same/ left and left outer join
select employee.id, employee.name, employee.deptid, deptid.name
from employee
left outer join deptid
on employee.deptid = deptid.id;

-- right join
select employee.id, employee.name, employee.deptid,deptid.id, deptid.name
from employee
right join deptid
on employee.deptid = deptid.id;

--full join--all the records are retrieved 
select employee.id, employee.name, employee.deptid, deptid.name
from employee
full join deptid
on employee.deptid = deptid.id;
--this doesnt work in mysql

--cross join, u dont need to specify join condition/ gives cartesian product
select *
from employee
cross join deptid
--not used in industry
--this is called full join in mysql

--natural joins doesn't work here
--it doesn't need join condition, it joins only if both column id matches (same column names)

/**15-Oct */

--assignment 3 discussion.

--self join

-- three table joins --refer pdf

-----------------------------------------------
/*22-Oct */

--Window functions
--we dont need multiple subqueries and multiple joins
--operates on a set of rows and return a single value for each row
--refer ppt for more details


--display names of emp, who is having max salary in a age group
select a.name , a.age , a.salary
from emp as a
inner join (
			select age,max(salary) salary 
			from emp group by age
			) as b 
on a.age=b.age and a.salary=b.salary

-- Now we dont need to do the extra joins
CREATE TABLE sales(
    sales_employee VARCHAR(50) , fiscal_year INT ,
    sale DECIMAL(14,2)
);
 
INSERT INTO sales(sales_employee,fiscal_year,sale)
VALUES
('Bob',2016,100), ('Bob',2017,150), ('Bob',2018,200),
('Alice',2016,150), ('Alice',2017,100), ('Alice',2018,200),
('John',2016,200), ('John',2017,150), ('John',2018,250);

Select * from sales

--group by vs window fn

select sum(sale) as total_sales, fiscal_year 
from sales
group by fiscal_year;


--window fn
select 
sales_employee , sale , fiscal_year,
sum(sale) over (partition by fiscal_year) as total_sales
from sales;

--here in above code, if you observe, the columns before "over" doesnt have a aggregate fn, but its there in o/p 
--whereas this is not possible in groupby.

SELECT  
	sales_employee
	, sale
	, fiscal_year
	, SUM(sale) OVER (PARTITION BY fiscal_year) sum_sale
	, max(sale) OVER (PARTITION BY fiscal_year) max_sale
	, Min(sale) OVER (PARTITION BY fiscal_year) Min_sale
	, Avg(sale) OVER (PARTITION BY fiscal_year) Avg_sale
FROM  
	sales; 

-- we dont need extra inner joins/ subqueries etc.,


select *
, sum(sale) over (partition by fiscal_year) as ssale 
, sum(sale) over (partition by fiscal_year order by fiscal_year) as Osale  
--order by here gives culmulative addition within the partition
from sales

select *
, sum(sale) over (partition by fiscal_year) as ssale 
, sum(sale) over (partition by fiscal_year order by fiscal_year) as Osale
, sum(sale) over (partition by fiscal_year order by sales_employee) as SEsale
--order by here gives culmulative addition within the partition
--here u can see tht clearly
from sales;


select *
, sum(sale) over (partition by fiscal_year) as ssale 
, sum(sale) over (partition by fiscal_year order by fiscal_year) as Osale
, sum(sale) over (partition by fiscal_year order by sales_employee) as SEsale
, sum(sale) over (partition by fiscal_year order by sales_employee 
 rows between  unbounded preceding and current row) as UCsale
 --now we have fixed the frame, till which row u need that cumulative prpty. 
from sales;

--, sum(sale) over (partition by fiscal_year order by sales_employee
-- rows between  unbounded preceding and 1 following) as csale
--, sum(sale) over (partition by fiscal_year order by sales_employee
-- rows between  1 preceding and 1 following) as fsale
--, sum(sale) over (partition by fiscal_year order by sales_employee) as Osale
--, sum(sale) over (partition by fiscal_year order by sale) as Osale

-------------------------------------------------------------------------------
/*26th oct*/

--More on window fn
--Dense rank, rank, rownumber

select * from sales;

--gives rank like we used to have in school
select *, RANK() over (order by sale) as RANK1
from sales;

--gives rank , without skipping.
select *, DENSE_RANK() over (order by sale) as DRANK2
from sales;

--gives unqiue row values
select *, ROW_NUMBER() over (order by sale) as Row_no
from sales;

select *, 
RANK() over (order by sale) as RANK1,
DENSE_RANK() over (order by sale) as DRANK2,
ROW_NUMBER() over (order by sale) as Row_no
from sales;

--question : Display the employee who is getting maximum salary within 
--                                                         a age group

select * from employee;

--nth highest from last
select T.name as "Employees who are receiving max_salary in particular age group"
from (select *,
		RANK() over (partition by age order by salary) as RANK1,
		ROW_NUMBER() over (partition by age order by salary desc) as R_no
		from employee) as T
where R_no=1
order by salary;

--Question : Display the people with 3rd highest salary

select t1.name as "Employee with 3rd highest rank"
from (select *, 
		DENSE_RANK() over(order by salary desc) as r
		from employee) as t1
where r =3

--without window fn
select name as "Employee with 3rd highest salary"
from employee 
where salary = (select min(salary) from employee
				where salary in 
				(select distinct top 3 salary from employee order by salary desc)
				)


--having 

select fiscal_year, max(sale) 
from sales
group by fiscal_year;

--difference between where and having
--where ---pre filter
select fiscal_year, max(sale) 
from sales
where sale >150
group by fiscal_year
-- this applies where filter then , it groups data and gives us the o/p

--same as above
select a.* from sales as a
inner join
(select fiscal_year fy, max(sale) msale from sales
group by fiscal_year) as b
on a.sale = b.msale and a.fiscal_year = b.fy
where b.msale >150

--having post filter
select fiscal_year, max(sale) 
from sales
group by fiscal_year, sale
having max(sale) >150 -- u can have aggregate on having filter 
--it applies to partition or only to the groupby 

--Interview Qns
/*
where applies on table; having applies on groups
having can use aggregate fn unlike where(in where we have to use subquery)
*/
/* Nxt session
date fn
constraints
e-r daig (normalisation)
*/
---------------------------------------------------------
/*29-Oct*/

--Date

select getdate() --get current date

select CURRENT_TIMESTAMP

--since getdate() will get the current timestamp, 
--if u r using it in code,always it is recommended to store it variables and use it.
Declare @d datetime
set @d = GETDATE()
select @d
waitfor delay '00:00:03'
select @d, getdate()

--extract the required part from date(in this case month)
select datepart(mm, '2020-10-29 09:17:40.003')

--difference between 2 dates

select getdate() - 10 --gives us 10days bfr tdy
select GETDATE() + 3 --adds 3 days
--we are adding or removing the no. of the days('day' not on month or year)

--benefit of using variables for datefns
--fn call is running only one time, so time is saved
declare @do datetime, @c datetime
set @do = GETDATE()

select @do
select @do -10
select @c = @do +3
select datediff(day, @do,@c) as DayDiff_var

select datediff(day, '2020-05-19 09:37:02.840','2020-11-01 09:37:02.840') as DayDiff
select datediff(mm, '2020-05-19 09:37:02.840','2020-11-01 09:37:02.840') as monthDiff


--select datediff(day, @do,@c) --added above with variable declaration

--add/subtract two dates
--dateadd(datepart, number, date)
--if number is negative it subtracts

declare @d2 date
set @d2 = GETDATE()
select DATEADD(day, 10, @d2) as newdate, @d2 as olddate

declare @d3 date
set @d3 = GETDATE()
select DATEADD(day, -20, @d3) as newdate, @d3 as olddate

--convert fn
--used to convert any dataype to new datatype
--syntax--convert(datatype, value, optinal parameter)

declare @a int, @b varchar(10)
set @a =1
set @b = convert(varchar(10), @a)
--sometimes this will fail --set @b = @a; so do conversion always
select @a, @b

declare @a int, @b varchar(100), @c datetime
set @a =1
set @c = getdate()
set @b = @c
--date can be converted to dtring, this has no issuses
/*set @a =@c --this wiil throw error*/
set @a = convert(int, @c)
select @a, @b, @c

--according to the optional parameter, it strips the date, changes the o/p
Declare @a datetime
Declare @b varchar (100) 
Set @a =getdate()
Select @a,
convert(varchar(30), @a),
convert(varchar(30), @a,100),
convert(varchar(30), @a,101),
convert(varchar(30), @a,102),
convert(varchar(30), @a,103),
convert(varchar(30), @a,104),
convert(varchar(30), @a,10)

--------------------------------
--constraints
/*
not null --no nul values accepted
null  -- can take null values
unique --no duplicates, data shld be unique(it can accept one null value)
primary key --used to uniquley identify each and every row(unique + not null) 
foreign key --to create a relationship between two tables
				The advantage of Foreign Keys is Referential Integrity. 
					This means that for every row in a Child table that has a foreign key, there will be a matching row in the Parent table.
check --if it satisfies the condition, data is inserted into the dable
default --defaultly inserts this value
*/

-------------------------------------------------------------
/*03-Nov*/

--more on foriegn key!! why we need it
--to avoid insert , update and delete anamolies

--Database diagram
--if there are 1000 tables how will you find the relationship, we can easily find relations using er diagram

--er diagram 
--to understand db relatonship
--rectangle, boxes, rhombus, circle etc.,
--refer pdf

--normalisation
https://www.edureka.co/blog/normalization-in-sql/#:~:text=Normalization%20entails%20organizing%20the%20columns,so%20it%20is%20more%20efficient.&text=Normalization%20in%20SQL%20will%20enhance%20the%20distribution%20of%20data.



