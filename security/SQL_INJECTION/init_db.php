<?php
$db = new SQLite3(__DIR__ . '/users.db');

$db->exec('DROP TABLE IF EXISTS users');
$db->exec('CREATE TABLE users (username TEXT NOT NULL, password TEXT NOT NULL)');

$demoUser = 'admin';
$demoPass = 'Admin@123';

$stmt = $db->prepare('INSERT INTO users (username, password) VALUES (:username, :password)');
$stmt->bindValue(':username', $demoUser, SQLITE3_TEXT);
$stmt->bindValue(':password', $demoPass, SQLITE3_TEXT);
$stmt->execute();

echo "Database initialized successfully.\n";
echo "Demo credentials: admin / Admin@123\n";
