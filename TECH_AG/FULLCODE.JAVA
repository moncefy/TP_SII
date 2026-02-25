package TECH_AG;

import java.util.ArrayList;
import java.util.Arrays;
public class FULLCODE {
//AAAA
 public static ArrayList<String> reste (ArrayList<String> a){
 ArrayList<String> r =new ArrayList<String>();
 for (int i = 1; i < a.size(); i++) {
 r.add(a.get(i));
 }
 return r;
 }
 //la fonction chain
 public static boolean chain(rule[] bdr , ArrayList<String> bdp,ArrayList<String>fp){
 boolean res;
 if(fp.isEmpty()){
 res=true;
 }else{
 if (dembut(bdr,bdp,fp.get(0))){
 bdp.add(fp.get(0));
 res=chain(bdr,bdp,reste(fp));
 }else{
 res=false;
 }
 }
 return res;
 }
 // dddd
 public static boolean dembut(rule[] bdr , ArrayList<String> bdp,String but){
 boolean res;
 if(bdp.contains(but)){
 res=true;
 }else{
 res=false;
 ArrayList<rule> RA = new ArrayList<rule>();
 //construction de l'ensemble des conflit
 for(int i=0 ; i< bdr.length;i++) {
 if (bdr[i].c.contains(but)) {
 RA.add(bdr[i]);
 }
 }
 while (!RA.isEmpty() && res!=true){
 //select
 rule r = RA.get(0);
 System.out.println(r.name+1);
 RA.remove(0);
 res=chain(bdr,bdp,r.p);
 }
 }
 return res;
 }
 public static void main(String[] args) {
 //create
 rule[] bdr= new rule[9];
 ArrayList<String> p = new ArrayList<>(Arrays.asList("A","B" ));
 ArrayList<String> c = new ArrayList<>(Arrays.asList("F"));
 bdr[0]=new rule(0,p,c);
 p = new ArrayList<>(Arrays.asList("F","H" ));
 c = new ArrayList<>(Arrays.asList("I"));
 bdr[1]=new rule(1,p,c);
 p = new ArrayList<>(Arrays.asList("D","H","G" ));
 c = new ArrayList<>(Arrays.asList("A"));
 bdr[2]=new rule(2,p,c);
 p = new ArrayList<>(Arrays.asList("O","G" ));
 c = new ArrayList<>(Arrays.asList("H"));
 bdr[3]=new rule(3,p,c);
 p = new ArrayList<>(Arrays.asList("E","H" ));
 c = new ArrayList<>(Arrays.asList("B" ));
 bdr[4]=new rule(4,p,c);
 p = new ArrayList<>(Arrays.asList("G","A" ));
 c = new ArrayList<>(Arrays.asList("B"));
 bdr[5]=new rule(5,p,c);
 p = new ArrayList<>(Arrays.asList("G","H" ));
 c = new ArrayList<>(Arrays.asList("P" ));
 bdr[6]=new rule(6,p,c);
 p = new ArrayList<>(Arrays.asList("G","H" ));
 c = new ArrayList<>(Arrays.asList("O" ));
 bdr[7]=new rule(7,p,c);
 p = new ArrayList<>(Arrays.asList("D","O","G" ));
 c = new ArrayList<>(Arrays.asList("J" ));
 bdr[8]=new rule(8,p,c);
 // definition
 ArrayList<String> fp = new ArrayList<String>( Arrays.asList("I"));
 //definition
 ArrayList<String> BDP = new ArrayList<String>(Arrays.asList("D","O","G" ));
 boolean test= chain(bdr,BDP,fp);
 // display
 System.out.println(test);
 }
}

// classe pour
class rule {
 boolean state; //
 ArrayList<String> p; // vvvvv
 ArrayList<String> c; //liste
 int name; // ggggg
 public rule(int i, ArrayList<String> a1, ArrayList<String> a2) {
 name = i;
 state = true;
 p = a1;
 c = a2;
 }
}