create database if not exists Tafelvoetbal;
use Tafelvoetbal;

create table if not exists Players (
    ID int not null auto_increment,
    Username varchar(255) not null,
    FirstName varchar(255),
    LastName varchar(255),
    Age int,
    Gender char(1),
    Occupation varchar(255),
    primary key (ID),
    unique (Username)
);

create table if not exists Games (
    ID int not null auto_increment,
    PlayerID_Black1 int not null,
    PlayerID_Black2 int,
    PlayerID_White1 int not null,
    PlayerID_White2 int,
    StartTime datetime not null,
    Duration time(4) not null,
    ScoreWhite tinyint not null,
    ScoreBlack tinyint not null,
    primary key (ID),
    foreign key (PlayerID_Black1) references Players(ID),
    foreign key (PlayerID_Black2) references Players(ID),
    foreign key (PlayerID_White1) references Players(ID),
    foreign key (PlayerID_White2) references Players(ID)
);

create table if not exists Hulls (
    ID int not null auto_increment,
    Hull varchar(8000),
    primary key (ID)
);

create table if not exists Datapoints (
    FrameNumber int not null,
    GameID int not null,
    HullID int not null,
    XCoord smallint,
    YCoord smallint,
    Speed float(53),
    Angle decimal(19,16),
    Accuracy decimal(19,18),
    HitType smallint default 0,
    foreign key (GameID) references Games(ID),
    foreign key (HullID) references Hulls(ID)
);

create table if not exists DatapointsML (
	GameID int not null,
    FrameNumber int not null,
    XCoord smallint,
    YCoord smallint,
    Speed float(53),
    Angle decimal(19,16),
    Accuracy decimal(19,18),
    HitType smallint default 0,
    SpeedHit boolean,
    AngleHit boolean
);