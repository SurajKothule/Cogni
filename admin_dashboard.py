#!/usr/bin/env python3
"""
Simple admin dashboard to view customer applications and statistics
"""

import os
import json
from pathlib import Path
from datetime import datetime
from customer_data.storage_manager import CustomerDataManager

class AdminDashboard:
    def __init__(self):
        self.storage_manager = CustomerDataManager()
    
    def display_stats(self):
        """Display statistics for all loan types"""
        print("📊 LOAN APPLICATION STATISTICS")
        print("=" * 50)
        
        loan_types = ["education", "home", "personal", "gold", "business", "car"]
        
        for loan_type in loan_types:
            stats = self.storage_manager.get_application_stats(loan_type)
            
            print(f"\n🎯 {loan_type.upper()} LOANS:")
            print(f"   Total Applications: {stats['total']}")
            print(f"   Completed: {stats['completed']}")
            print(f"   Approved: {stats['approved']}")
            print(f"   Partial Approval: {stats['partial']}")
            
            if stats['completed'] > 0:
                print(f"   Average Eligible Amount: ₹{stats['average_amount']:,.0f}")
                print(f"   Average Interest Rate: {stats['average_interest']:.2f}%")
    
    def display_recent_applications(self, loan_type: str, limit: int = 5):
        """Display recent applications for a loan type"""
        print(f"\n📋 RECENT {loan_type.upper()} APPLICATIONS (Last {limit})")
        print("=" * 60)
        
        applications = self.storage_manager.get_customer_applications(loan_type, limit)
        
        if not applications:
            print("No applications found.")
            return
        
        for i, app in enumerate(applications, 1):
            customer = app["customer_info"]
            timestamp = datetime.fromisoformat(app["timestamp"])
            
            print(f"\n{i}. Application ID: {app['session_id'][:8]}...")
            print(f"   Date: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Customer: {customer['name']}")
            print(f"   Email: {customer['email']}")
            print(f"   Phone: {customer['phone']}")
            print(f"   Status: {app['status']}")
            
            if app.get("prediction_result"):
                result = app["prediction_result"]["result"]
                print(f"   Eligible Amount: ₹{result['eligible_amount']:,}")
                print(f"   Interest Rate: {result['interest_rate']}%")
                print(f"   Approval Status: {result['status']}")
    
    def export_to_csv(self, loan_type: str):
        """Show CSV file location"""
        csv_path = Path("customer_data") / loan_type / "reports" / f"{loan_type}_applications.csv"
        
        if csv_path.exists():
            print(f"\n📄 CSV Export available at: {csv_path}")
            print(f"   File size: {csv_path.stat().st_size} bytes")
            print(f"   Last modified: {datetime.fromtimestamp(csv_path.stat().st_mtime)}")
        else:
            print(f"\n❌ No CSV file found for {loan_type} loans")
    
    def interactive_menu(self):
        """Interactive menu for admin operations"""
        while True:
            print("\n" + "="*50)
            print("🏦 LOAN ADMIN DASHBOARD")
            print("="*50)
            print("1. View Overall Statistics")
            print("2. View Recent Education Loan Applications")
            print("3. View Recent Home Loan Applications") 
            print("4. View Recent Personal Loan Applications")
            print("5. View Recent Gold Loan Applications")
            print("6. View Recent Business Loan Applications")
            print("7. View Recent Car Loan Applications")
            print("8. Export CSV Reports")
            print("9. Exit")
            
            choice = input("\nSelect an option (1-9): ").strip()
            
            if choice == "1":
                self.display_stats()
            
            elif choice == "2":
                limit = input("Number of applications to show (default 5): ").strip()
                limit = int(limit) if limit.isdigit() else 5
                self.display_recent_applications("education", limit)
            
            elif choice == "3":
                limit = input("Number of applications to show (default 5): ").strip()
                limit = int(limit) if limit.isdigit() else 5
                self.display_recent_applications("home", limit)
            
            elif choice == "4":
                limit = input("Number of applications to show (default 5): ").strip()
                limit = int(limit) if limit.isdigit() else 5
                self.display_recent_applications("personal", limit)
            
            elif choice == "5":
                limit = input("Number of applications to show (default 5): ").strip()
                limit = int(limit) if limit.isdigit() else 5
                self.display_recent_applications("gold", limit)
            
            elif choice == "6":
                limit = input("Number of applications to show (default 5): ").strip()
                limit = int(limit) if limit.isdigit() else 5
                self.display_recent_applications("business", limit)
            
            elif choice == "7":
                limit = input("Number of applications to show (default 5): ").strip()
                limit = int(limit) if limit.isdigit() else 5
                self.display_recent_applications("car", limit)
            
            elif choice == "8":
                print("\n📊 CSV EXPORT LOCATIONS:")
                for loan_type in ["education", "home", "personal", "gold", "business", "car"]:
                    self.export_to_csv(loan_type)
            
            elif choice == "9":
                print("\n👋 Goodbye!")
                break
            
            else:
                print("\n❌ Invalid option. Please try again.")

def main():
    dashboard = AdminDashboard()
    
    # Check if any data exists
    data_dir = Path("customer_data")
    if not data_dir.exists():
        print("❌ No customer data directory found.")
        print("💡 Start the loan application system first to generate data.")
        return
    
    dashboard.interactive_menu()

if __name__ == "__main__":
    main()