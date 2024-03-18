#include <string.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <tuple>
#include <fstream>
#include <typeinfo>
#include <math.h>
#include <random>

using namespace std;
typedef double T;



bool contains(char arr[], string s){
    char c = s[0];
    bool contain = false;
    for (int i=0; i<strlen(arr);i++){
        if (c==arr[i]){
            contain = true;
            break;
        }
    }
    return contain;
}

char* ops(string i, char* variables, char* operations_1, char* operations_2){
    if (i=="0") {return variables;}
    if (i=="1") {return operations_1;}
    if (i=="2") {return operations_2;}
};

vector<int> each_ids(int flat_id, vector<int> prod_dims){
    int len = prod_dims.size();
    vector<int> each_id;
    for (int i=0;i<len;i++){
        int de = prod_dims[len-i-1];
        int q = flat_id/de;
        flat_id = flat_id - q*de;
        each_id.push_back(q);
    }
    return each_id;
}


T H_evaluate(vector<char> RPN, int arity_len, vector<T> z){
    vector<T> stacks;
    int j = 0;
    for (int k=0; k<arity_len; k++){
    char sym = RPN[k];
    if (sym=='+') {
        stacks[j-2] = stacks[j-2]+stacks[j-1];
        j = j-1;
        stacks.pop_back();
        }
    if (sym=='-') {
        stacks[j-2] = stacks[j-2]-stacks[j-1];
        j = j-1;
        stacks.pop_back();
        }
    if (sym=='*') {
        stacks[j-2] = stacks[j-2]*stacks[j-1];
        j = j-1;
        stacks.pop_back();
        }
    if (sym=='%') {
        if (int(stacks[j-1]) == 0){
            stacks[j-2] = 1000.;
        } else {
        stacks[j-2] = int(stacks[j-2]) % int(stacks[j-1]);
        }
        j = j-1;
        stacks.pop_back();
        }
    if (sym=='>') {
         stacks[j-1] = stacks[j-1] + 1;
        }
    if (sym=='<') {
         stacks[j-1] = stacks[j-1] - 1;
        }
    if (sym=='~') {
         stacks[j-1] = - stacks[j-1];
        }
    if (sym=='H') {
         stacks[j-1] = stacks[j-1] > 0;
        }
    if (sym=='D') {
         stacks[j-1] = stacks[j-1] == 0;
        }
    if (sym=='A') {
         stacks[j-1] = abs(stacks[j-1]);
        }
    if (sym=='U') {
         stacks[j-1] = stacks[j-1] + 100;
        }
    if (sym=='V') {
         stacks[j-1] = stacks[j-1] - 100;
        }
    if (sym=='a') {
         stacks.push_back(z[0]);
         j = j+1;
        }
    if (sym=='b') {
         stacks.push_back(z[1]);
         j = j+1;
        }
    if (sym=='c') {
         stacks.push_back(z[2]);
         j = j+1;
        }
    if (sym=='d') {
         stacks.push_back(z[3]);
         j = j+1;
        }
    if (sym=='e') {
         stacks.push_back(z[4]);
         j = j+1;
        }
    if (sym=='f') {
         stacks.push_back(z[5]);
         j = j+1;
        }
    if (sym=='g') {
         stacks.push_back(z[6]);
         j = j+1;
        }
    if (sym=='h') {
         stacks.push_back(z[7]);
         j = j+1;
        }
    if (sym=='i') {
         stacks.push_back(z[8]);
         j = j+1;
        }
    if (sym=='j') {
         stacks.push_back(z[9]);
         j = j+1;
        }
    if (sym=='k') {
         stacks.push_back(z[10]);
         j = j+1;
        }
    if (sym=='1') {
         stacks.push_back(1.0);
         j = j+1;
        }
    if (sym=='2') {
         stacks.push_back(2.0);
         j = j+1;
        }
    if (sym=='3') {
         stacks.push_back(3.0);
         j = j+1;
        }
    if (sym=='4') {
         stacks.push_back(4.0);
         j = j+1;
        }
    if (sym=='5') {
         stacks.push_back(5.0);
         j = j+1;
        }
    if (sym=='6') {
         stacks.push_back(6.0);
         j = j+1;
        }
    if (sym=='7') {
         stacks.push_back(7.0);
         j = j+1;
        }
    if (sym=='8') {
         stacks.push_back(8.0);
         j = j+1;
        }
    if (sym=='9') {
         stacks.push_back(9.0);
         j = j+1;
        }
    }
    return stacks[0];
}



T norm(vector<T> vec, int dim){
    T squared = 0.0;
    for (int i=0;i<dim;i++){
        squared = squared + pow(vec[i],2);
    }
    return sqrt(squared);

}vector<T> normalized_vec(vector<T> vec, int dim){
    T norm_vec = norm(vec, dim);
    vector<T> vecp = vec;
    if (norm_vec<1e-12) {
    for (int i=0;i<dim;i++){
        vecp[i] = 1.0;
        /*throw 20;*/
    }
    norm_vec = norm(vecp, dim);
    }
    for (int i=0;i<dim;i++){
        vecp[i] = vecp[i]/norm_vec;
    }
    return vecp;
}


void printrpn(vector<char> RPN){
    int num = RPN.size();
    for (int i=0;i<num;i++){
        cout << RPN[i];
    }
    cout << endl;
}

//int main(){
int main(int argc, char* argv[]){
    string env = argv[1]; //"rnn_unique2_numerical"; //argv[1];
    string target = argv[2]; //argv[2]; //argv[1]; //"test"; // pass to program
    const int input_dim = atoi(argv[3]); //atoi(argv[3]); // 2; // pass to program
    int test_pts = 25; //25;
    /*char variables2[] = {'a','b','c','d','e','f','g','h','i','j','k','\0'};
    char variables[input_dim+1];
    if (input_dim > 11){cout << "Support <= 11 input variables!";};
    for (int i=0;i<input_dim;i++){
        variables[i] = variables2[i];
    }
    variables[input_dim] = '\0';*/
    
    const int num_const = 2;
    char variables2[] = {'a','b','c','d','e','f','g','h','i','j','k','\0'};
    char consts[] = {'3','4','5','6','7','8','9','\0'};
    char variables[input_dim+1+num_const];
    if (input_dim > 11){cout << "Support <= 11 input variables!";};
    for (int i=0;i<input_dim;i++){
        variables[i] = variables2[i];
    }
    for (int i=0;i<num_const;i++){
        variables[input_dim+i] = consts[i];
    }
    variables[input_dim+num_const] = '\0';
    //char variables[] = {'a','b','c','d','e','\0'};
    // cout << variables[0] << variables[1] << variables[2] << variables[3] << variables[4] << variables[5] << variables[6] << endl;
    // Note: the char array should end with '\0'
    char operations_1[] = {'>','<','~','H','D','A','U','V','\0'};
    char operations_2[] = {'+','*','-','%','\0'};
    ifstream input("./tasks/"+env+"/"+target+".txt");
    vector<vector<T>> data_var;
    int test_pts_id = 0;
    for (string line; getline(input,line);)
    {
        if (test_pts_id == test_pts){break;}
        vector<T> dp_var;
        size_t pos = 0;
        string delimiter = " ";
        string token;
        int i = 0;
        while ((pos = line.find(delimiter)) != string::npos) {
            token = line.substr(0, pos);
            if (i < input_dim+1){
                dp_var.push_back(stof(token));
                /*if (i == input_dim-1){
                    for (int i=0;i<num_const;i++){
                        dp_var.push_back(i+1.0);
                    }
                }*/
            }
            line.erase(0, pos + delimiter.length());
            i = i + 1;
        }
        data_var.push_back(dp_var);
        test_pts_id = test_pts_id + 1;
    }
    /*read arity file*/
    ifstream input2("./symbolic_regression/arity2templates.txt");
    vector<string> arities;
    /*number of templates tried*/
    int num_templates = 38; //38;//89;
    int template_id = 0;
    for (string line; getline(input2,line);)
    {
        if (template_id == num_templates){break;}
        arities.push_back(line);
        template_id = template_id + 1;
    }
    
    ofstream myfile;
    myfile.open ("./tasks/"+env+"/"+target+"_symbolic.txt");
    myfile.close();
    
    cout << "symbolic regressing" << endl;
    
    for (int k=0;k<num_templates;k++){

        string arity = arities[k]; //"00202"
        //cout << "k=" << k << "," << "arity=" << arity << endl;
        /*string arity = "01012";*/
        int arity_len = arity.length();

        vector<int> dims;
        vector<int> prod_dims;
        vector<string> symbol_types;
        int flat_size = 1;
        for (int i=0; i<arity.length();i++){
            string symbol_type = arity.substr(i,1);
            int dim = strlen(ops(symbol_type, variables, operations_1,operations_2));
            //cout << dim;
            symbol_types.push_back(symbol_type);
            dims.push_back(dim);
            prod_dims.push_back(flat_size);
            flat_size = flat_size * dim;
        }
        //cout << endl << flat_size << endl;

        for (int i=0; i<flat_size;i++){
            vector<int> each_id = each_ids(i, prod_dims);
            vector<char> RPN;
            for (int j=0;j<arity_len;j++){
                char* ops_ = ops(arity.substr(j,1), variables, operations_1, operations_2);
                char symbol = ops_[each_id[arity_len-1-j]];
                RPN.push_back(symbol);
            }
            
            //vector<char> RPN = {'a','b','+','3','m'};
            //vector<char> RPN = {'a','b','+'};
            //cout << i << endl;
            //printrpn(RPN);
            T eps = 0.0;
            bool flag = true;
            for (int j=0; j<test_pts; j++){
                T eq_value = H_evaluate(RPN, arity_len, {data_var[j].begin(), data_var[j].end() - 1});
                //cout << eq_value << ',' << data_var[j][input_dim] << endl;
                eps = eps + pow(eq_value - data_var[j][input_dim],2);
            }
            eps = eps/test_pts;
            //cout << eps << endl;
            if (eps<0.0001){
                string RPN_string(RPN.begin(), RPN.end());
                //cout << RPN_string + " " + to_string(eps) <<endl ;
                ofstream myfile;
                myfile.open ("./tasks/"+env+"/"+target+"_symbolic.txt", std::ios_base::app);
                myfile << RPN_string + " " + to_string(eps) <<endl;
                myfile.close();
                return 0;
            }
        }
    }
    return 0;
}
