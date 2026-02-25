package TECH_AG;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

public class ForwardChainingApp extends JFrame {
    private JTextArea factsArea, rulesArea, resultsArea;
    private JTextField goalField;
    private JButton startButton, clearButton;
    private ForwardChaining fc;

    public ForwardChainingApp() {
        setTitle("Forward Chaining - Knowledge-Based System");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1000, 700);
        setLocationRelativeTo(null);
        setLayout(new GridLayout(2, 1));

        // Top Panel - Input
        JPanel inputPanel = new JPanel(new GridLayout(1, 2, 10, 10));
        inputPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        // Left side - Facts and Rules
        JPanel leftPanel = new JPanel(new BorderLayout(5, 5));
        leftPanel.setBorder(BorderFactory.createTitledBorder("Knowledge Base"));

        JPanel factsPanel = new JPanel(new BorderLayout(5, 5));
        factsPanel.setBorder(BorderFactory.createTitledBorder("Facts (comma-separated)"));
        factsArea = new JTextArea(6, 30);
        factsArea.setText("D, O, G");
        factsArea.setFont(new Font("Monospaced", Font.PLAIN, 12));
        factsPanel.add(new JScrollPane(factsArea), BorderLayout.CENTER);

        JPanel rulesPanel = new JPanel(new BorderLayout(5, 5));
        rulesPanel.setBorder(BorderFactory.createTitledBorder("Rules (format: P1,P2->C)"));
        rulesArea = new JTextArea(8, 30);
        rulesArea.setText("A,B->F\nF,H->I\nD,H,G->A\nO,G->H\nE,H->B\nG,A->B\nG,H->P\nG,H->O\nD,O,G->J");
        rulesArea.setFont(new Font("Monospaced", Font.PLAIN, 12));
        rulesPanel.add(new JScrollPane(rulesArea), BorderLayout.CENTER);

        leftPanel.add(factsPanel, BorderLayout.NORTH);
        leftPanel.add(rulesPanel, BorderLayout.CENTER);

        // Right side - Goal and Results
        JPanel rightPanel = new JPanel(new BorderLayout(5, 5));
        rightPanel.setBorder(BorderFactory.createTitledBorder("Inference Engine"));

        JPanel goalPanel = new JPanel(new BorderLayout(5, 5));
        goalPanel.setBorder(BorderFactory.createTitledBorder("Target Goal"));
        goalField = new JTextField("I");
        goalField.setFont(new Font("Monospaced", Font.PLAIN, 14));
        goalPanel.add(goalField, BorderLayout.CENTER);

        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 5));
        startButton = new JButton("Start Forward Chaining");
        clearButton = new JButton("Clear Results");
        buttonPanel.add(startButton);
        buttonPanel.add(clearButton);

        JPanel resultsPanel = new JPanel(new BorderLayout(5, 5));
        resultsPanel.setBorder(BorderFactory.createTitledBorder("Results & Trace"));
        resultsArea = new JTextArea(10, 30);
        resultsArea.setEditable(false);
        resultsArea.setFont(new Font("Monospaced", Font.PLAIN, 11));
        resultsPanel.add(new JScrollPane(resultsArea), BorderLayout.CENTER);

        rightPanel.add(goalPanel, BorderLayout.NORTH);
        rightPanel.add(buttonPanel, BorderLayout.CENTER);
        rightPanel.add(resultsPanel, BorderLayout.SOUTH);

        inputPanel.add(leftPanel);
        inputPanel.add(rightPanel);

        add(inputPanel);

        // Button Listeners
        startButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                runForwardChaining();
            }
        });

        clearButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                resultsArea.setText("");
            }
        });

        setVisible(true);
    }

    private void runForwardChaining() {
        try {
            // Parse facts
            Set<String> facts = parseFacts(factsArea.getText());
            
            // Parse rules
            ArrayList<Rule> rules = parseRules(rulesArea.getText());
            
            // Get goal
            String goal = goalField.getText().trim().toUpperCase();
            
            if (facts.isEmpty() || rules.isEmpty()) {
                resultsArea.setText("Error: Facts and Rules cannot be empty!");
                return;
            }

            // Run forward chaining
            fc = new ForwardChaining(facts, rules);
            boolean success = fc.infer(goal);

            // Display results
            StringBuilder result = new StringBuilder();
            result.append("=== FORWARD CHAINING RESULTS ===\n\n");
            result.append("Initial Facts: ").append(facts).append("\n");
            result.append("Goal: ").append(goal).append("\n");
            result.append("Rules Count: ").append(rules.size()).append("\n\n");
            result.append("=== INFERENCE TRACE ===\n");
            result.append(fc.getTrace());
            result.append("\n=== FINAL RESULTS ===\n");
            result.append("All Facts Derived: ").append(fc.getAllFacts()).append("\n");
            result.append("Goal [" + goal + "] Found: ").append(success ? "✓ YES" : "✗ NO").append("\n");

            resultsArea.setText(result.toString());
        } catch (Exception e) {
            resultsArea.setText("Error: " + e.getMessage());
        }
    }

    private Set<String> parseFacts(String text) {
        Set<String> facts = new HashSet<>();
        String[] arr = text.split(",");
        for (String f : arr) {
            String fact = f.trim().toUpperCase();
            if (!fact.isEmpty()) {
                facts.add(fact);
            }
        }
        return facts;
    }

    private ArrayList<Rule> parseRules(String text) {
        ArrayList<Rule> rules = new ArrayList<>();
        String[] lines = text.split("\n");
        
        for (String line : lines) {
            line = line.trim();
            if (line.isEmpty()) continue;
            
            if (!line.contains("->")) {
                throw new IllegalArgumentException("Invalid rule format: " + line);
            }
            
            String[] parts = line.split("->");
            String[] premises = parts[0].split(",");
            String[] conclusions = parts[1].split(",");
            
            ArrayList<String> p = new ArrayList<>();
            for (String premise : premises) {
                String pr = premise.trim().toUpperCase();
                if (!pr.isEmpty()) p.add(pr);
            }
            
            ArrayList<String> c = new ArrayList<>();
            for (String conclusion : conclusions) {
                String con = conclusion.trim().toUpperCase();
                if (!con.isEmpty()) c.add(con);
            }
            
            if (!p.isEmpty() && !c.isEmpty()) {
                rules.add(new Rule(rules.size(), p, c));
            }
        }
        return rules;
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                new ForwardChainingApp();
            }
        });
    }
}

class ForwardChaining {
    private Set<String> facts;
    private ArrayList<Rule> rules;
    private StringBuilder trace;

    public ForwardChaining(Set<String> initialFacts, ArrayList<Rule> rules) {
        this.facts = new HashSet<>(initialFacts);
        this.rules = rules;
        this.trace = new StringBuilder();
    }

    public boolean infer(String goal) {
        int iteration = 0;
        boolean foundNewFact = true;

        trace.append("Starting with facts: ").append(facts).append("\n\n");

        while (foundNewFact) {
            iteration++;
            foundNewFact = false;
            trace.append("[Iteration ").append(iteration).append("]\n");

            for (Rule rule : rules) {
                if (canApplyRule(rule)) {
                    // Apply rule
                    for (String conclusion : rule.getConclusions()) {
                        if (!facts.contains(conclusion)) {
                            facts.add(conclusion);
                            foundNewFact = true;
                            trace.append("  ✓ Rule ").append(rule.getName())
                                    .append(": ").append(rule.getPremises())
                                    .append(" → ").append(conclusion).append("\n");

                            // Check if goal is reached
                            if (conclusion.equals(goal)) {
                                trace.append("\n✓ GOAL [").append(goal).append("] FOUND!\n");
                                return true;
                            }
                        }
                    }
                }
            }

            if (!foundNewFact) {
                trace.append("  No new facts derived. Stopping.\n");
            } else {
                trace.append("  Current facts: ").append(facts).append("\n\n");
            }
        }

        // Check if goal is in final facts
        if (facts.contains(goal)) {
            trace.append("\n✓ GOAL [").append(goal).append("] FOUND!\n");
            return true;
        } else {
            trace.append("\n✗ GOAL [").append(goal).append("] NOT FOUND!\n");
            return false;
        }
    }

    private boolean canApplyRule(Rule rule) {
        for (String premise : rule.getPremises()) {
            if (!facts.contains(premise)) {
                return false;
            }
        }
        return true;
    }

    public String getTrace() {
        return trace.toString();
    }

    public Set<String> getAllFacts() {
        return facts;
    }
}

class Rule {
    private int name;
    private ArrayList<String> premises;
    private ArrayList<String> conclusions;

    public Rule(int name, ArrayList<String> premises, ArrayList<String> conclusions) {
        this.name = name;
        this.premises = new ArrayList<>(premises);
        this.conclusions = new ArrayList<>(conclusions);
    }

    public int getName() {
        return name;
    }

    public ArrayList<String> getPremises() {
        return premises;
    }

    public ArrayList<String> getConclusions() {
        return conclusions;
    }

    @Override
    public String toString() {
        return premises + " → " + conclusions;
    }
}
