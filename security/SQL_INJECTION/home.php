<?php
session_start();

if (!isset($_SESSION['logged_in']) || $_SESSION['logged_in'] !== true) {
    header('Location: index.php');
    exit;
}

$username = $_SESSION['username'] ?? 'Unknown user';
?>
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Home - SQL Injection Demo</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <main class="container">
        <section class="card">
            <h1>Home Page</h1>
            <p>Welcome, <strong><?php echo htmlspecialchars($username, ENT_QUOTES, 'UTF-8'); ?></strong>!</p>
            <p>This page is intentionally protected by a vulnerable login form for educational demonstration.</p>

            <a class="button-link" href="logout.php">Logout</a>
        </section>
    </main>
</body>
</html>
