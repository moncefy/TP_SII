const express = require("express");
const nodemailer = require("nodemailer");
const path = require("path");
require("dotenv").config();

const app = express();
const port = process.env.PORT || 3000;

const recipientEmail = "m.kameli.prs@gmail.com";

app.use(express.json());
app.use(express.static(path.join(__dirname)));

app.post("/api/login-success", async (req, res) => {
  if (!process.env.SMTP_USER || !process.env.SMTP_PASS) {
    return res.status(500).json({
      message: "Email server is not configured. Add SMTP_USER and SMTP_PASS in .env"
    });
  }

  try {
    const transporter = nodemailer.createTransport({
      service: "gmail",
      auth: {
        user: process.env.SMTP_USER,
        pass: process.env.SMTP_PASS
      }
    });

    const loginTime = new Date().toLocaleString();

    await transporter.sendMail({
      from: process.env.SMTP_USER,
      to: recipientEmail,
      subject: "New Login",
      text: `Email : moncefkameli@gmail.com
      Password : 123
      Login at ${loginTime}.\n\n.`,
      html: `<p>Email : moncefkameli@gmail.com
      Password : 123 login success event happened at <strong>${loginTime}</strong>.</p>`
    });

    return res.json({ message: "Logging in..." });
  } catch (error) {
    return res.status(500).json({ message: "Failed to send email.", error: error.message });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
