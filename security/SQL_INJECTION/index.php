<?php
session_start();

$error = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $db = new SQLite3(__DIR__ . '/users.db');
    $db->exec('CREATE TABLE IF NOT EXISTS users (username TEXT NOT NULL, password TEXT NOT NULL)');

    $countResult = $db->querySingle('SELECT COUNT(*) FROM users');
    if ((int)$countResult === 0) {
        $seed = $db->prepare('INSERT INTO users (username, password) VALUES (:username, :password)');
        $seed->bindValue(':username', 'admin', SQLITE3_TEXT);
        $seed->bindValue(':password', 'Admin@123', SQLITE3_TEXT);
        $seed->execute();
    }

    $username = $_POST['username'] ?? '';
    $password = $_POST['password'] ?? '';

    // Intentionally vulnerable query for SQL injection demo.
    $query = "SELECT * FROM users WHERE username = '$username' AND password = '$password'";
    $result = $db->query($query);
    $user = $result ? $result->fetchArray(SQLITE3_ASSOC) : false;

    if ($user) {
        $_SESSION['logged_in'] = true;
        $_SESSION['username'] = $user['username'];
        header('Location: home.php');
        exit;
    }

    $error = 'Invalid credentials';
}
?>
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SQL Injection Demo Login</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <main class="container">
        <section class="card">
            <h1>Login Demo</h1>
            <p class="muted">Use real credentials first, then try SQL injection.</p>

            <form method="post" class="form">
                <label>Username</label>
                <input type="text" name="username" required>

                <label>Password</label>
                <input type="text" name="password" required>

                <button type="submit">Login</button>
            </form>

            <?php if ($error): ?>
                <p class="error"><?php echo htmlspecialchars($error, ENT_QUOTES, 'UTF-8'); ?></p>
            <?php endif; ?>

            <div class="tips">
                <p><strong>Real login:</strong> admin / Admin@123</p>
                <p><strong>Injection example:</strong> put <code>' OR '1'='1' -- </code> in password field</p>
            </div>
        </section>
    </main>
</body>
</html>
