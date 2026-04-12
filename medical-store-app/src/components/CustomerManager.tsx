import React, { useState } from 'react';
import { Plus, Edit2, Trash2, Phone, Mail } from 'lucide-react';
import { Customer } from '../types';
import { getCustomers, saveCustomer, getBills } from '../storage';
import '../styles/CustomerManager.css';

export default function CustomerManager() {
  const [customers, setCustomers] = useState<Customer[]>(getCustomers());
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [formData, setFormData] = useState<Partial<Customer>>({
    name: '',
    phone: '',
    email: '',
    address: '',
  });

  const bills = getBills();

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!formData.name || !formData.phone) {
      alert('Please fill required fields (Name, Phone)');
      return;
    }

    if (editingId) {
      const customer: Customer = {
        id: editingId,
        name: formData.name,
        phone: formData.phone,
        email: formData.email,
        address: formData.address,
        registeredDate: customers.find(c => c.id === editingId)?.registeredDate || new Date().toISOString().split('T')[0],
        totalPurchases: customers.find(c => c.id === editingId)?.totalPurchases || 0,
      };
      saveCustomer(customer);
    } else {
      const newCustomer: Customer = {
        id: `CUST-${Date.now()}`,
        name: formData.name,
        phone: formData.phone,
        email: formData.email || '',
        address: formData.address || '',
        registeredDate: new Date().toISOString().split('T')[0],
        totalPurchases: 0,
      };
      saveCustomer(newCustomer);
    }

    setCustomers(getCustomers());
    resetForm();
  };

  const resetForm = () => {
    setFormData({ name: '', phone: '', email: '', address: '' });
    setEditingId(null);
    setShowForm(false);
  };

  const handleEdit = (customer: Customer) => {
    setFormData(customer);
    setEditingId(customer.id);
    setShowForm(true);
  };

  const handleDelete = (id: string) => {
    if (confirm('Are you sure? This cannot be undone.')) {
      setCustomers(customers.filter(c => c.id !== id));
    }
  };

  const filteredCustomers = customers.filter(c =>
    c.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    c.phone.includes(searchTerm)
  );

  const getCustomerSpent = (customerId: string) => {
    return bills
      .filter(b => b.customerId === customerId)
      .reduce((sum, b) => sum + b.totalAmount, 0);
  };

  return (
    <div className="customer-manager">
      <div className="customer-header">
        <h2 className="card-title">Customer Management</h2>
        <button className="btn-primary button" onClick={() => setShowForm(true)}>
          <Plus size={18} /> Add Customer
        </button>
      </div>

      {/* Stats */}
      <div className="grid-3">
        <div className="stat-card">
          <div className="stat-label">Total Customers</div>
          <div className="stat-value">{customers.length}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Avg. Customer Spend</div>
          <div className="stat-value">
            ₹{customers.length > 0
              ? (customers.reduce((sum, c) => sum + getCustomerSpent(c.id), 0) / customers.length).toFixed(0)
              : 0}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Repeat Customers</div>
          <div className="stat-value">
            {customers.filter(c => getCustomerSpent(c.id) > 0).length}
          </div>
        </div>
      </div>

      {/* Form Modal */}
      {showForm && (
        <div className="modal-overlay" onClick={() => resetForm()}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <h3 className="modal-title">
              {editingId ? 'Edit Customer' : 'Add New Customer'}
            </h3>

            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label className="form-label">Name *</label>
                <input
                  type="text"
                  name="name"
                  className="form-input"
                  value={formData.name || ''}
                  onChange={handleInputChange}
                  required
                />
              </div>

              <div className="form-group">
                <label className="form-label">Phone *</label>
                <input
                  type="tel"
                  name="phone"
                  className="form-input"
                  value={formData.phone || ''}
                  onChange={handleInputChange}
                  required
                />
              </div>

              <div className="form-group">
                <label className="form-label">Email</label>
                <input
                  type="email"
                  name="email"
                  className="form-input"
                  value={formData.email || ''}
                  onChange={handleInputChange}
                />
              </div>

              <div className="form-group">
                <label className="form-label">Address</label>
                <textarea
                  name="address"
                  className="form-textarea"
                  value={formData.address || ''}
                  onChange={handleInputChange}
                  rows={3}
                />
              </div>

              <div className="modal-footer">
                <button type="button" className="btn-secondary button" onClick={resetForm}>
                  Cancel
                </button>
                <button type="submit" className="btn-primary button">
                  {editingId ? 'Update' : 'Add'} Customer
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Search */}
      <div className="card">
        <input
          type="text"
          className="search-input"
          placeholder="Search by name or phone..."
          value={searchTerm}
          onChange={e => setSearchTerm(e.target.value)}
        />
      </div>

      {/* Customers List */}
      <div className="card">
        {filteredCustomers.length === 0 ? (
          <p className="empty-state">No customers found</p>
        ) : (
          <div className="customers-grid">
            {filteredCustomers.map(customer => {
              const spent = getCustomerSpent(customer.id);
              return (
                <div key={customer.id} className="customer-card">
                  <div className="customer-header-card">
                    <h3>{customer.name}</h3>
                    <span className="customer-badge">{customer.totalPurchases} orders</span>
                  </div>

                  <div className="customer-contact">
                    <div className="contact-item">
                      <Phone size={16} />
                      <span>{customer.phone}</span>
                    </div>
                    {customer.email && (
                      <div className="contact-item">
                        <Mail size={16} />
                        <span>{customer.email}</span>
                      </div>
                    )}
                  </div>

                  {customer.address && (
                    <div className="customer-address">
                      <strong>Address:</strong> {customer.address}
                    </div>
                  )}

                  <div className="customer-stats">
                    <div className="stat">
                      <span className="stat-label">Total Spent</span>
                      <span className="stat-value">₹{spent.toLocaleString()}</span>
                    </div>
                    <div className="stat">
                      <span className="stat-label">Registered</span>
                      <span className="stat-value">{customer.registeredDate}</span>
                    </div>
                  </div>

                  <div className="customer-actions">
                    <button
                      className="btn-secondary button"
                      onClick={() => handleEdit(customer)}
                    >
                      <Edit2 size={16} /> Edit
                    </button>
                    <button
                      className="btn-danger button"
                      onClick={() => handleDelete(customer.id)}
                    >
                      <Trash2 size={16} /> Delete
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
