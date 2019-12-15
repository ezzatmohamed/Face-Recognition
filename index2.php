<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta http-equiv="X-UA-Compatible" content="ie=edge" />
    <link
      rel="stylesheet"
      href="https://use.fontawesome.com/releases/v5.7.2/css/all.css"
      integrity="sha384-fnmOCqbTlWIlj8LyTjo7mOUStjsKC4pOpQbqyi7RrhN7udi9RwhKkMHpvLbHG9Sr"
      crossorigin="anonymous"
    />
    <link
      rel="stylesheet"
      href="https://bootswatch.com/4/journal/bootstrap.min.css"
    />
    <title>IP Face Detection & Recognition</title>
  </head>
  <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
    <a class="navbar-brand" href="#">Face Detection & Recognition Dashboard </a>
    <button
      class="navbar-toggler"
      type="button"
      data-toggle="collapse"
      data-target="#navbarColor01"
      aria-controls="navbarColor01"
      aria-expanded="false"
      aria-label="Toggle navigation"
    >
      <span class="navbar-toggler-icon"></span>
    </button>

    <div class="collapse navbar-collapse" id="navbarColor01">
      <ul class="navbar-nav mr-auto">
        <li class="nav-item active">
          <a class="nav-link" href="/Dashbord.html"
            >Home <span class="sr-only">(current)</span></a
          >
        </li>
    
      </ul>
    </div>
  </nav>

  <div id="myTabContent" class="tab-content">
    <div class="tab-pane fade show active" id="home">
      <div class="jumbotron">
        <h1 class="display-3">Hello !</h1>
        <p class="lead">
          This is a Face Detection and Recognition project for Image Processing
          Course
        </p>
        <hr class="my-4" />
        <p>
          It uses HOG Classifier for Face Detection and Skin Segmentation BLA
          BLA. For Face Recognition, it uses LBP Algorithm and classifies Faces
          accordingly.
        </p>
        <p class="lead">
          <a
            class="btn btn-primary btn-lg"
            href="https://github.com/ezzatmohamed/Face-Recognition/tree/master"
            role="button"
            >Learn more</a
          >
        </p>
        <form method="GET" action="index.php">

          <input name="ip" type="text" placeholder="Enter IP"><br><br>

          <input role="button"  class="btn btn-primary btn-lg" type="submit" value = "Test With Mobile Stream">

        </form>
        <br>
        <br>
        

        <form method="GET" action="index.php">

          <input name="name" type="text" placeholder="Enter Filename"><br><br>

          <input role="button"  class="btn btn-primary btn-lg" type="submit" value = "Test With Video">

        </form><br><br>
        <p class="lead">
          <a 
            class="btn btn-primary btn-lg"
            href="index.php?web=True"
            role="button"
            >Test With Webcam</a
          >
        </p>
      </div>
    </div>
</html>
<?php

if (isset($_GET['ip']))
{
    $ip = $_GET['ip'];
    $out = shell_exec("python3 Recognition.py 1 ".$ip );
    header('Location: index2.php');
    die();
}
else if ( $_GET['web'] == "True")
{
  $out = shell_exec("python3 Recognition.py 3");
  header('Location: index2.php');
  die();
}
else if (isset($_GET['name']) )
{
  $out = shell_exec("python3 Recognition.py 2 ".$_GET['name']);
  header('Location: index2.php');
  die();
}
?>