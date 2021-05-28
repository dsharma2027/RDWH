Use [Praxis July_2020]

-- ASSIGNMENT 1-----

-- Creating a table Employee--
create table employee
(	Id int,
	Name VARCHAR(200),
	Department varchar(100),
	Join_date Date,
	Age smallint,
	Salary bigint,
	City Varchar(100),
	Country Varchar(100)
)

select * from employee

-- Inserting a single entry at a time--
Insert into employee
values (1, 'Ankit', 'Finance', '2015-10-13', 24, 48000, 'Delhi','India');

select * from employee

Insert into employee
values 
(2, 'Bhavya', 'HR', '2017-09-23', 29, 51000, 'Mumbai','India')

Insert into employee
values
(3, 'Chinmay', 'Operations', '2012-12-01', 32, 49000, 'Mumbai','India')

Insert into	employee
values
(4, 'Deeksha', 'Finance', '2019-02-10', 28, 35000, 'Pune','India')

Insert into employee
values
(5, 'Gautam', 'Analyst', '2011-11-09', 36, 160000, 'New York','USA')

Insert into employee
Values
(6, 'Harshit', 'Sales', '2009-11-30', 27, 55000, 'Kolkata','India')

-- Inserting a Multiple entry at once--
Insert into employee
values
(7, 'Isha', 'HR', '2014-05-20', 38, 45000, 'Delhi','India'),
(8, 'Monika', 'Operations', '2020-01-13', 21, 28000, 'Pune','India'),
(9, 'Nitin', 'Analyst', '2016-07-22', 28, 95000, 'London','UK'),
(10, 'Rahul', 'Sales', '2012-12-12', 30, 40000, 'Mumbai','India');




select * from employee

--Selecting the rows where Department is Finance--
select * from employee
where Department = 'Finance'

--Selecting the rows where Country is not India--
select * from employee
where Not Country = 'India'

--Selecting the rows where Salary > 50000--
Select * from employee
where Salary > 50000

--Selecting the rows where Department is HR and Age>30--
Select * from employee
where Department = 'HR' and Age > 30

Select * from employee
where Join_date > '2015-01-01'

-- Adding the new column Gender--
alter table employee
add Gendre char(10)

select * from employee

--- Changing the Variable size

alter table employee
alter column Name varchar(500)


-- Droping the column gender from Table--
alter table employee
drop column Gendre

alter table employee
add Gender char(10)

-- Mistakenly Created the New rows while updating the Gendre column--
Insert into employee (Gender)
Values
('M'), ('F'), ('M'), ('F'), ('M'), ('M'), ('F'), ('F'), ('M'), ('M')

select * from employee

-- Deleting the newly created rows--
Delete from employee
where Gender = 'M' or Gender = 'F'

Select * from employee


-- Deleting all the elements of the Table Employee
Truncate table employee

Select * from employee

----------------------------------------------------------------------------------------------------------------------------

--- ASSIGNMENT 2--------

-- Extract File name from the File Destination.

Declare @addr varchar (1000)
Set @addr = 'C:\SQL2019\Express_ENU\PackageId.dat'


SELECT LEFT(@addr,LEN(@addr) - charindex('\',reverse(@addr),1) + 1) AS p_name, 
       RIGHT(@addr, CHARINDEX('\', REVERSE(@addr))-1)  AS f_name

SELECT LEFT(@addr,LEN(@addr) - charindex('\',reverse(@addr)) + 1) AS p_name, 
       RIGHT(@addr, CHARINDEX('\', REVERSE(@addr))-1)  AS f_name


---------------------------------------------------------------------------------------------------------------------------

--ASSIGNMENT 3--

-- Output ofthe Below Query--

--select firstname, lastname, id, salary from emp
--where salary = (
--				 Select min(salary) from emp
--				 where salary > (select avg(salary) from emp
--								  where age = ( select top 1 age from emp	
--												where  salary = (select max(salary) from emp)
--												)
--								)	
--				and age = (select top 1 age from  emp where salary = (select  max(salary) from emp) )
				
--			and age = (select top 1 age  from emp where Salary = (select  max(salary) from emp))
			)

-- Answer: Print the Firstname, last name, id and salary of the emp who has minimum salary out of the employee of  with max salary for each age group..

---------------------------------------------------------------------------------------------------------------------------------------------------------------------
